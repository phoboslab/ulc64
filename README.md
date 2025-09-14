Quick test using [ulc-codec](https://github.com/Aikku93/ulc-codec) on the N64 with [libdragon](https://github.com/DragonMinded/libdragon) (preview branch).

The decoder runs entirely on the CPU. It's realtime, albeit barely.

Demo songs in 16, 32, 64 and 128 kbps are included. Just change `"rom://demo-32.ulc"` to `"rom://demo-64.ulc"` etc. in `ulcdemo.c` to try.