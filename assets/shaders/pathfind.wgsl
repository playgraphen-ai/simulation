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
};

struct PathRequest {
    start: u32,
    target_seg: u32,
    person_id: u32,
};

struct PathRequestQueue {
    count_x: u32,
    count_y: u32,
    count_z: u32,
    pad: u32,
    requests: array<PathRequest>,
};

@group(0) @binding(0) var roads_tex  : texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var<storage, read_write> prev: array<i32>;
@group(0) @binding(2) var<storage, read_write> paths: array<u32>;
@group(0) @binding(3) var<uniform> params: PathParams;
@group(0) @binding(4) var<storage, read> path_queue: PathRequestQueue;

fn seg_coords(seg: u32) -> array<vec2<i32>, 2> {
    let base = i32(seg * 2u);
    let w = i32(params.roads_tex_w);
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    return array<vec2<i32>, 2>(c0, c1);
}

// Conn packing: count in bits 24..31; three ids in bits 0..23 (8 bits each).
fn unpack_conn(packed: f32) -> array<u32, 4> {
    let p = bitcast<u32>(packed);
    let count = (p >> 24u) & 0xFFu;
    let a = (p >> 16u) & 0xFFu;
    let b = (p >> 8u) & 0xFFu;
    let c = p & 0xFFu;
    return array<u32, 4>(count, a, b, c);
}

const MAX_FRONTIER: u32 = 1024u;
var<workgroup> frontier: array<u32, 1024>;
var<workgroup> frontier_next: array<u32, 1024>;
var<workgroup> frontier_len: atomic<u32>;
var<workgroup> frontier_next_len: atomic<u32>;
var<workgroup> found: atomic<u32>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let req_id = wg.x;
    if req_id >= path_queue.count_x { return; }
    let req = path_queue.requests[req_id];

    let slice = params.segments_count;
    let base_prev = req_id * slice;

    // Initialize prev to -1 for this request's segments.
    var init_idx = lidx;
    while init_idx < slice {
        prev[base_prev + init_idx] = -1;
        init_idx = init_idx + 64u;
    }
    
    if lidx == 0u {
        atomicStore(&frontier_len, 1u);
        atomicStore(&frontier_next_len, 0u);
        if req.start == req.target_seg {
            atomicStore(&found, 1u);
        } else {
            atomicStore(&found, 0u);
        }
        frontier[0] = req.start;
        prev[base_prev + req.start] = i32(req.start);
    }
    workgroupBarrier();

    // Expand frontier.
    for (var level: u32 = 0u; level < params.max_path_len; level = level + 1u) {
        let is_found = atomicLoad(&found);
        let flen = atomicLoad(&frontier_len);
        
        var i = lidx;
        while i < flen && is_found == 0u {
            let cur = frontier[i];
            let coords = seg_coords(cur);
            let t1 = textureLoad(roads_tex, coords[1]);
            let cap = unpack_conn(t1.y);
            let cbp = unpack_conn(t1.z);
            let total_count = cap[0] + cbp[0];

            for (var n: u32 = 0u; n < 6u; n = n + 1u) {
                var nb: u32;
                if n < cap[0] { nb = cap[n + 1u]; }
                else if n < total_count { nb = cbp[n - cap[0] + 1u]; }
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
            frontier[j] = frontier_next[j];
            j = j + 64u;
        }
        
        workgroupBarrier();
        
        // We can't break early safely in D3D12 if it bypasses workgroupBarrier.
        // But since we removed breaks, we just spin idly if found == 1u or nlen == 0u.
    }

    // Reconstruct path.
    if lidx == 0u {
        let base_path = req.person_id * params.max_path_len;
        // Initialize path to sentinel (u32::MAX).
        for (var pidx: u32 = 0u; pidx < params.max_path_len; pidx = pidx + 1u) {
            paths[base_path + pidx] = 0xFFFFFFFFu;
        }

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
        }
    }
}
