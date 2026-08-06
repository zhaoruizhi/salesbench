"""Pre-extract frames for all videos to speed up VLM baseline runs."""
import sys
import json
import time

sys.path.insert(0, "src")

from salesbench.vlm.frame_sampler import sample_frames

def main():
    with open("input/raw_video/video_index.jsonl") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    success = 0
    fail = 0
    skip_no_video = 0
    start = time.time()

    print(f"Pre-extracting frames for {total} videos...")

    for idx, rec in enumerate(records):
        video_id = rec.get("video_id", "")
        video_path = rec.get("primary_video_path", "")
        has_video = rec.get("has_video_asset", 0)

        if not video_path or not has_video:
            skip_no_video += 1
            continue

        try:
            frames = sample_frames(
                video_path=video_path,
                video_id=video_id,
                strategy="hook_plus_uniform",
                total_frames=8,
            )
            if frames:
                success += 1
            else:
                fail += 1
                print(f"  ⚠ {video_id}: no frames extracted")
        except Exception as e:
            fail += 1
            print(f"  ✗ {video_id}: {e}")

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            eta = (total - idx - 1) / rate
            print(f"  [{idx+1}/{total}] success={success} fail={fail} skip={skip_no_video} ETA={eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Success: {success}")
    print(f"  Failed:  {fail}")
    print(f"  No video: {skip_no_video}")

if __name__ == "__main__":
    main()
