// Acts like a game loading its textures: takes graphics memory in 256 MiB steps until only
// STOP_AT MiB are free, holds it for HOLD seconds, then gives it all back.
// It never goes below 2048 MiB free (the desktop's share, see ../../CLAUDE.md).
//   vram-eater STOP_AT_MiB HOLD_SECONDS
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>
#include <glob.h>

// Free VRAM as the kernel counts it (sysfs), on the card with the most VRAM. hipMemGetInfo reported
// ~700 MiB more free than this on 2026-10-28, so it can't be trusted for the 2 GiB line.
static long sysfs_free_mib() {
    glob_t g;
    long best_total = 0, best_free = -1;
    if (glob("/sys/class/drm/card*/device/mem_info_vram_total", 0, nullptr, &g) != 0) return -1;
    for (size_t i = 0; i < g.gl_pathc; i++) {
        std::string t = g.gl_pathv[i], u = t.substr(0, t.size() - 5) + "used";
        FILE * ft = fopen(t.c_str(), "r"), * fu = fopen(u.c_str(), "r");
        long long total = 0, used = 0;
        if (ft && fu && fscanf(ft, "%lld", &total) == 1 && fscanf(fu, "%lld", &used) == 1 && total > best_total) {
            best_total = total;
            best_free = (long)((total - used) >> 20);
        }
        if (ft) fclose(ft);
        if (fu) fclose(fu);
    }
    globfree(&g);
    return best_free;
}

int main(int argc, char ** argv) {
    long stop_at = argc > 1 ? atol(argv[1]) : 3072;
    int hold = argc > 2 ? atoi(argv[2]) : 20;
    if (stop_at < 2048) stop_at = 2048;
    const size_t step = 256ull << 20;
    std::vector<void *> held;
    for (;;) {
        long free_mib = sysfs_free_mib();
        if (free_mib < 0) { fprintf(stderr, "eater: can't read free VRAM from sysfs\n"); break; }
        if (free_mib - 256 < stop_at) { printf("eater: holding %zu MiB, %ld MiB free\n", held.size() * 256, free_mib); break; }
        void * p = nullptr;
        if (hipMalloc(&p, step) != hipSuccess) { printf("eater: allocation refused at %ld MiB free\n", free_mib); break; }
        (void)hipMemset(p, 0x5a, step);         // touch it, like real textures
        (void)hipDeviceSynchronize();
        held.push_back(p);
        usleep(300 * 1000);
    }
    fflush(stdout);
    sleep(hold);
    for (void * p : held) (void)hipFree(p);
    printf("eater: gave back %zu MiB\n", held.size() * 256);
    return 0;
}
