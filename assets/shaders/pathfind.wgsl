// Compute shader: batched segment-level BFS pathfinding.
// Each person needing a path is assigned to one workgroup.
//
// Bindings:
//   @group(0) @binding(0) var roads_tex  : texture_storage_2d<rgba32float, read>;
//   @group(0) @binding(1) var<storage, read_write> prev: array<i32>;
//   @group(0) @binding(2) var<storage, read_write> paths: array<u32>;  // MAX_PATH_LEN per request
//   @group(0) @binding(3) var<uniform> params: PathParams;
//   @group(0) @binding(4) var<storage, read> requests: array<PathRequest>;

struct PathParams {
    segments_count: u32,
    roads_tex_w: u32,
    max_path_len: u32,
    request_count: u32,
    slice_start: u32,
    slice_end: u32,
    do_dispatch: u32,
    reset_path_queue: u32,
};

struct PathRequest {
    start: u32,
    target_seg: u32,
    person_id: u32,
    pad: u32,
};

struct PathRequestQueue {
    count_x: atomic<u32>,
    count_y: u32,
    count_z: u32,
    processed: atomic<u32>,
    requests: array<PathRequest>,
};

@group(0) @binding(0) var roads_tex  : texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var<storage, read_write> prev: array<i32>;
@group(0) @binding(2) var<storage, read_write> paths: array<u32>;
@group(0) @binding(3) var<uniform> params: PathParams;
@group(0) @binding(4) var<storage, read_write> path_queue: PathRequestQueue;

fn seg_coords(seg: u32) -> array<vec2<i32>, 4> {
    let base = seg * 4u;
    let w = max(1u, params.roads_tex_w);
    let c0 = vec2<i32>(i32(base % w), i32(base / w));
    let c1 = vec2<i32>(i32((base + 1u) % w), i32((base + 1u) / w));
    let c2 = vec2<i32>(i32((base + 2u) % w), i32((base + 2u) / w));
    let c3 = vec2<i32>(i32((base + 3u) % w), i32((base + 3u) / w));
    return array<vec2<i32>, 4>(c0, c1, c2, c3);
}

const MAX_FRONTIER: u32 = 1024u;
var<workgroup> frontier: array<u32, 1024>;
var<workgroup> frontier_next: array<u32, 1024>;
var<workgroup> frontier_len: atomic<u32>;
var<workgroup> frontier_next_len: atomic<u32>;
var<workgroup> found: atomic<u32>;
var<workgroup> shared_req_id: u32;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    if lidx == 0u {
        shared_req_id = atomicAdd(&path_queue.processed, 1u);
    }
    workgroupBarrier();
    let req_id = shared_req_id;
    let max_req = min(atomicLoad(&path_queue.count_x), 16384u);

    // D3D12/FXC compiler complains if we return early inside a workgroup barrier loop
    // To solve this, we don't return early. We use a boolean flag to wrap all operations.
    let is_valid_req = req_id < max_req;
    var req: PathRequest;
    if is_valid_req {
        req = path_queue.requests[req_id];
    }

    let slice = params.segments_count;
    let safe_req_id = min(req_id, 16383u);
    let base_prev = safe_req_id * slice;

    // Initialize prev to -1 for this request's segments.
    var init_idx = lidx;
    while init_idx < slice {
        if is_valid_req {
            prev[base_prev + init_idx] = -1;
        }
        init_idx = init_idx + 64u;
    }
    
    if lidx == 0u {
        atomicStore(&frontier_len, 1u);
        atomicStore(&frontier_next_len, 0u);
        if is_valid_req {
            if req.start == req.target_seg {
                atomicStore(&found, 1u);
            } else {
                atomicStore(&found, 0u);
            }
            frontier[0] = req.start;
            prev[base_prev + req.start] = i32(req.start);
        }
    }
    workgroupBarrier();

    // Expand frontier.
    for (var level: u32 = 0u; level < params.max_path_len; level = level + 1u) {
        let is_found = atomicLoad(&found);
        let flen = atomicLoad(&frontier_len);
        
        var i = lidx;
        while i < flen && is_found == 0u {
            if is_valid_req {
                let cur = frontier[i];
                let coords = seg_coords(cur);
                let t2 = textureLoad(roads_tex, coords[2]);
                let t3 = textureLoad(roads_tex, coords[3]);
                
                let count_a = u32(t2.w);
                let count_b = u32(t3.w);
                let total_count = count_a + count_b;

                for (var n: u32 = 0u; n < 6u; n = n + 1u) {
                    var nb: u32;
                    if n < count_a { 
                        if (n == 0u) { nb = u32(t2.x); }
                        else if (n == 1u) { nb = u32(t2.y); }
                        else { nb = u32(t2.z); }
                    }
                    else if n < total_count { 
                        let bn = n - count_a;
                        if (bn == 0u) { nb = u32(t3.x); }
                        else if (bn == 1u) { nb = u32(t3.y); }
                        else { nb = u32(t3.z); }
                    }
                    else { continue; }
                    
                    if nb >= params.segments_count { continue; }

                    if prev[base_prev + nb] < 0 {
                        prev[base_prev + nb] = i32(cur);
                        if nb == req.target_seg {
                            atomicStore(&found, 1u);
                        }
                        let next_idx = atomicAdd(&frontier_next_len, 1u);
                        if next_idx < MAX_FRONTIER {
                            frontier_next[next_idx] = nb;
                        }
                    }
                }
            }
            i = i + 64u;
        }
        
        workgroupBarrier();
        
        let nlen = atomicLoad(&frontier_next_len);
        
        if lidx == 0u {
            atomicStore(&frontier_len, min(nlen, MAX_FRONTIER));
            atomicStore(&frontier_next_len, 0u);
        }
        
        workgroupBarrier();
        
        var j = lidx;
        while j < atomicLoad(&frontier_len) && atomicLoad(&found) == 0u {
            if is_valid_req {
                frontier[j] = frontier_next[j];
            }
            j = j + 64u;
        }
        
        workgroupBarrier();
        
        // We can't break early safely in D3D12 if it bypasses workgroupBarrier.
        // But since we removed breaks, we just spin idly if found == 1u or nlen == 0u.
    }

    // Reconstruct path.
    if lidx == 0u && is_valid_req {
        let safe_person_id = min(req.person_id, 65535u);
        let base_path = safe_person_id * params.max_path_len;

        if atomicLoad(&found) == 1u {
            var cur = req.target_seg;
            var path_tmp: array<u32, 256>; // Local temporary storage for reversal
            var count: u32 = 0u;

            while count < params.max_path_len {
                path_tmp[count] = cur;
                count = count + 1u;
                let p = prev[base_prev + cur];
                if p < 0 || u32(p) == cur { break; }
                cur = u32(p);
            }

            // Store path in forward order.
            for (var k: u32 = 0u; k < count; k = k + 1u) {
                paths[base_path + k] = path_tmp[count - 1u - k];
            }

            // Pad the rest of the path buffer with the sentinel value
            for (var k: u32 = count; k < params.max_path_len; k = k + 1u) {
                paths[base_path + k] = 0xFFFFFFFFu;
            }
        } else {
             // Path not found, set the first element to sentinel so the car fails gracefully.
             paths[base_path] = 0xFFFFFFFFu;
        }
    }
    }
