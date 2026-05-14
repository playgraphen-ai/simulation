// Compute shader: batched segment-level A* pathfinding with fixed memory window.
// Each person needing a path is assigned to one workgroup.

struct PathParams {
    segments_count: u32,
    roads_tex_w: u32,
    max_path_len: u32,
    request_count: u32,
    slice_start: u32,
    slice_end: u32,
    do_dispatch: u32,
    reset_path_queue: u32,
    major_segments_count: u32,
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

struct MajorRoadRow {
    ax: f32, ay: f32,
    bx: f32, by: f32,
    speed_mean: f32,
    length: f32,
    original_id: u32,
    count_a: u32,
    count_b: u32,
    conn_a: array<i32, 4>,
    conn_b: array<i32, 4>,
    pad: array<u32, 2>,
};

@group(0) @binding(0) var roads_tex  : texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var<storage, read_write> prev: array<atomic<u32>>; // Hash Table: 512 entries per request
@group(0) @binding(2) var<storage, read_write> paths: array<u32>;
@group(0) @binding(3) var<uniform> params: PathParams;
@group(0) @binding(4) var<storage, read_write> path_queue: PathRequestQueue;
@group(0) @binding(5) var<storage, read> major_graph: array<MajorRoadRow>;

fn hash_u32(x: u32) -> u32 {
    var v = x;
    v = ((v >> 16u) ^ v) * 0x45d9f3bu;
    v = ((v >> 16u) ^ v) * 0x45d9f3bu;
    v = (v >> 16u) ^ v;
    return v;
}

fn set_prev_ht(base: u32, key: u32, val: u32) -> bool {
    let k16 = key & 0xFFFFu;
    let v16 = val & 0xFFFFu;
    let new_entry = (k16 << 16u) | v16;
    
    var h = hash_u32(key) % 512u;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) { 
        let slot = base + ((h + i) % 512u);
        let res = atomicCompareExchangeWeak(&prev[slot], 0xFFFFFFFFu, new_entry);
        if res.exchanged {
            return true;
        }
        if (res.old_value >> 16u) == k16 {
            return false; // Already exists
        }
    }
    return false; // Table full or collision limit
}

fn get_prev_ht(base: u32, key: u32) -> i32 {
    let k16 = key & 0xFFFFu;
    var h = hash_u32(key) % 512u;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        let slot = base + ((h + i) % 512u);
        let entry = atomicLoad(&prev[slot]);
        if entry == 0xFFFFFFFFu { return -1; }
        if (entry >> 16u) == k16 {
            return i32(entry & 0xFFFFu);
        }
    }
    return -1;
}

fn seg_coords(seg: u32) -> array<vec2<i32>, 5> {
    let base = seg * 5u;
    let w = max(1u, params.roads_tex_w);
    let c0 = vec2<i32>(i32(base % w), i32(base / w));
    let c1 = vec2<i32>(i32((base + 1u) % w), i32((base + 1u) / w));
    let c2 = vec2<i32>(i32((base + 2u) % w), i32((base + 2u) / w));
    let c3 = vec2<i32>(i32((base + 3u) % w), i32((base + 3u) / w));
    let c4 = vec2<i32>(i32((base + 4u) % w), i32((base + 4u) / w));
    return array<vec2<i32>, 5>(c0, c1, c2, c3, c4);
}

const MAX_OPEN: u32 = 1024u;
var<workgroup> open_data: array<u32, 1024>;
var<workgroup> open_g: array<f32, 1024>;
var<workgroup> next_data: array<u32, 1024>;
var<workgroup> next_g: array<f32, 1024>;
var<workgroup> next_len: atomic<u32>;
var<workgroup> found: atomic<u32>;
var<workgroup> shared_v: u32;
var<workgroup> shared_req_id: u32;

var<workgroup> best_seg: u32;
var<workgroup> min_h: f32;
var<workgroup> shared_use_hierarchical: u32;

fn pack_f(f: f32, id: u32) -> u32 {
    let f_u = u32(clamp(f * 100.0, 0.0, 524287.0));
    return (f_u << 13) | (id & 0x1FFFu);
}

fn unpack_id(data: u32) -> u32 {
    return data & 0x1FFFu;
}

fn sort_open(lidx: u32) {
    for (var k: u32 = 2u; k <= 1024u; k = k << 1u) {
        for (var j: u32 = k >> 1u; j > 0u; j = j >> 1u) {
            for (var c: u32 = 0u; c < 8u; c = c + 1u) {
                let idx = lidx * 8u + c;
                let ixj = idx ^ j;
                if (ixj > idx) {
                    let da = open_data[idx];
                    let db = open_data[ixj];
                    let asc = (idx & k) == 0u;
                    if ((asc && da > db) || (!asc && da < db)) {
                        open_data[idx] = db;
                        open_data[ixj] = da;
                        let ga = open_g[idx];
                        open_g[idx] = open_g[ixj];
                        open_g[ixj] = ga;
                    }
                }
            }
            workgroupBarrier();
        }
    }
}

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
    let max_req = min(atomicLoad(&path_queue.count_x), 131072u);

    let is_valid_req = req_id < max_req;
    var req: PathRequest;
    if is_valid_req {
        req = path_queue.requests[req_id];
    }

    // Each request has exactly 512 entries in prev.
    let base_prev = req_id * 512u;

    // Initialize only our window of 512 entries.
    var init_idx = lidx;
    while init_idx < 512u {
        if is_valid_req {
            atomicStore(&prev[base_prev + init_idx], 0xFFFFFFFFu);
        }
        init_idx = init_idx + 64u;
    }

    var target_pos: vec2<f32>;
    var start_pos: vec2<f32>;
    if is_valid_req {
        let target_coords = seg_coords(req.target_seg);
        let target_t0 = textureLoad(roads_tex, target_coords[0]);
        target_pos = target_t0.xy;

        let start_coords = seg_coords(req.start);
        let start_t0 = textureLoad(roads_tex, start_coords[0]);
        start_pos = start_t0.xy;
    }

    // --- Hierarchical Layer (Détour Rapide) ---
    if lidx == 0u {
        var use_h = 0u;
        if is_valid_req && params.major_segments_count > 0u {
            let dist = distance(start_pos, target_pos);
            if dist > 250.0 { // threshold for "long" trip
                use_h = 1u;
            }
        }
        shared_use_hierarchical = use_h;
    }
    workgroupBarrier();

    let use_hierarchical = shared_use_hierarchical != 0u;

    if use_hierarchical {
        // Find nearest major segments
        var start_m: i32 = -1;
        var target_m: i32 = -1;
        var min_d_s = 1e10;
        var min_d_t = 1e10;

        // Simpler: Thread 0 does the search. It's only up to ~1024 segments, very fast on GPU.
        if lidx == 0u {
            for (var i: u32 = 0u; i < params.major_segments_count; i = i + 1u) {
                let m = major_graph[i];
                let d_s = min(distance(start_pos, vec2<f32>(m.ax, m.ay)), distance(start_pos, vec2<f32>(m.bx, m.by)));
                let d_t = min(distance(target_pos, vec2<f32>(m.ax, m.ay)), distance(target_pos, vec2<f32>(m.bx, m.by)));
                if d_s < min_d_s { min_d_s = d_s; start_m = i32(i); }
                if d_t < min_d_t { min_d_t = d_t; target_m = i32(i); }
            }
            
            // If they are the same or not found, fall back.
            if start_m == -1 || target_m == -1 || start_m == target_m {
                found = 0u; // Fallback
            } else {
                // Initialize Major A*
                found = 0u;
                best_seg = u32(start_m);
                min_h = distance(vec2<f32>(major_graph[start_m].ax, major_graph[start_m].ay), target_pos);
                open_data[0] = pack_f(min_h, u32(start_m));
                open_g[0] = 0.0;
                set_prev_ht(base_prev, u32(start_m), u32(start_m));
                for (var i: u32 = 1u; i < 1024u; i = i + 1u) { open_data[i] = 0xFFFFFFFFu; }
            }
        }
        workgroupBarrier();

        // Major A* Iterations
        for (var step: u32 = 0u; step < 256u; step = step + 1u) {
            sort_open(lidx);
            if found != 0u || open_data[0] == 0xFFFFFFFFu { break; }
            if lidx == 0u { atomicStore(&next_len, 0u); }
            workgroupBarrier();

            let cur_packed = open_data[lidx];
            if is_valid_req && cur_packed != 0xFFFFFFFFu {
                let cur = unpack_id(cur_packed);
                let cur_g = open_g[lidx];
                open_data[lidx] = 0xFFFFFFFFu;

                let m = major_graph[cur];
                let total_count = m.count_a + m.count_b;
                for (var n: u32 = 0u; n < 8u; n = n + 1u) {
                    var nb: i32 = -1;
                    if n < m.count_a { nb = m.conn_a[n]; }
                    else if n < total_count { nb = m.conn_b[n - m.count_a]; }
                    if nb == -1 { continue; }

                    if set_prev_ht(base_prev, u32(nb), cur) {
                        let n_m = major_graph[nb];
                        let nb_h = distance(vec2<f32>(n_m.ax, n_m.ay), target_pos);
                        if nb == target_m { found = 1u; }
                        let nb_cost = n_m.length / max(0.1, n_m.speed_mean);
                        let nb_g = cur_g + nb_cost;
                        let nb_f = nb_g + nb_h;
                        let slot = atomicAdd(&next_len, 1u);
                        if slot < 1024u {
                            next_data[slot] = pack_f(nb_f, u32(nb));
                            next_g[slot] = nb_g;
                        }
                    }
                }
            }
            workgroupBarrier();
            
            if lidx == 0u {
                let nlen = min(atomicLoad(&next_len), 1024u);
                for (var i: u32 = 0u; i < nlen; i = i + 1u) {
                    let id = unpack_id(next_data[i]);
                    let n_m = major_graph[id];
                    let h = distance(vec2<f32>(n_m.ax, n_m.ay), target_pos);
                    if h < min_h { min_h = h; best_seg = id; }
                }
                var v: u32 = 0u;
                while v < 1024u && open_data[v] != 0xFFFFFFFFu { v = v + 1u; }
                shared_v = v;
            }
            workgroupBarrier();
            let nlen = atomicLoad(&next_len);
            let v = shared_v;
            var i = lidx;
            while i < nlen {
                var slot: u32;
                if i < 64u { slot = i; } else { slot = v + (i - 64u); }
                if slot < 1024u { open_data[slot] = next_data[i]; open_g[slot] = next_g[i]; }
                i = i + 64u;
            }
            workgroupBarrier();
        }

        // Reconstruct Hierarchical Path
        if lidx == 0u {
            let safe_person_id = min(req.person_id, 524287u);
            let base_path = safe_person_id * params.max_path_len;
            var cur = best_seg;
            var path_tmp: array<u32, 512>;
            var count: u32 = 0u;
            
            // Final segment is the target_seg itself
            path_tmp[count] = req.target_seg;
            count = count + 1u;

            while count < params.max_path_len {
                path_tmp[count] = major_graph[cur].original_id;
                count = count + 1u;
                let p = get_prev_ht(base_prev, cur);
                if p < 0 || u32(p) == cur { break; }
                cur = u32(p);
            }
            
            // Add start segment if space
            if count < params.max_path_len {
                path_tmp[count] = req.start;
                count = count + 1u;
            }

            for (var k: u32 = 0u; k < count; k = k + 1u) {
                paths[base_path + k] = path_tmp[count - 1u - k];
            }
            for (var k: u32 = count; k < params.max_path_len; k = k + 1u) {
                paths[base_path + k] = 0xFFFFFFFFu;
            }
        }
    } else {
        // --- Regular A* (Step 3 or Fallback) ---
        if lidx == 0u {
            if is_valid_req {
                if req.start == req.target_seg {
                    atomicStore(&found, 1u);
                } else {
                    atomicStore(&found, 0u);
                }
                let h = distance(start_pos, target_pos);
                
                best_seg = req.start;
                min_h = h;

                open_data[0] = pack_f(h, req.start);
                open_g[0] = 0.0;
                set_prev_ht(base_prev, req.start, req.start);
            } else {
                open_data[0] = 0xFFFFFFFFu;
            }
            for (var i: u32 = 1u; i < 1024u; i = i + 1u) {
                open_data[i] = 0xFFFFFFFFu;
            }
        }
        workgroupBarrier();

        // A* iterations. Window is 512, but we can iterate more if neighbors overlap.
        // However, set_prev_ht will fail once the table is full (512 unique nodes).
        for (var step: u32 = 0u; step < params.max_path_len; step = step + 1u) {
            sort_open(lidx);

            let is_found = atomicLoad(&found);
            if is_found != 0u { break; }
            if open_data[0] == 0xFFFFFFFFu { break; } // Open set empty

            if lidx == 0u { atomicStore(&next_len, 0u); }
            workgroupBarrier();

            // Expand top 64 nodes in parallel.
            let cur_packed = open_data[lidx];
            if is_valid_req && cur_packed != 0xFFFFFFFFu {
                let cur = unpack_id(cur_packed);
                let cur_g = open_g[lidx];

                // Mark as processed in the open set
                open_data[lidx] = 0xFFFFFFFFu;

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

                    if set_prev_ht(base_prev, nb, cur) {
                        let nb_coords = seg_coords(nb);
                        let nb_t0 = textureLoad(roads_tex, nb_coords[0]);
                        let nb_t1 = textureLoad(roads_tex, nb_coords[1]);
                        
                        let nb_h = distance(nb_t0.xy, target_pos);
                        
                        if nb == req.target_seg {
                            atomicStore(&found, 1u);
                        }

                        let nb_cost = nb_t1.w / max(0.1, nb_t1.x);
                        let nb_g = cur_g + nb_cost;
                        let nb_f = nb_g + nb_h;

                        let slot = atomicAdd(&next_len, 1u);
                        if slot < 1024u {
                            next_data[slot] = pack_f(nb_f, nb);
                            next_g[slot] = nb_g;
                        }
                    }
                }
            }
            workgroupBarrier();

            // Update best_seg if any new node is closer.
            // We do this by having lidx 0 scan the newly added nodes in next_data.
            if lidx == 0u && is_valid_req {
                let nlen = min(atomicLoad(&next_len), 1024u);
                for (var i: u32 = 0u; i < nlen; i = i + 1u) {
                    let id = unpack_id(next_data[i]);
                    let nb_coords = seg_coords(id);
                    let nb_t0 = textureLoad(roads_tex, nb_coords[0]);
                    let h = distance(nb_t0.xy, target_pos);
                    if h < min_h {
                        min_h = h;
                        best_seg = id;
                    }
                }
            }
            workgroupBarrier();

            // Merge next into open set.
            if lidx == 0u {
                var v: u32 = 0u;
                while v < 1024u && open_data[v] != 0xFFFFFFFFu {
                    v = v + 1u;
                }
                shared_v = v;
            }
            workgroupBarrier();

            let nlen = atomicLoad(&next_len);
            let v = shared_v;

            var i = lidx;
            while i < nlen {
                var slot: u32;
                if i < 64u {
                    slot = i; // Reuse the slots we just expanded
                } else {
                    slot = v + (i - 64u);
                }
                if slot < 1024u {
                    open_data[slot] = next_data[i];
                    open_g[slot] = next_g[i];
                }
                i = i + 64u;
            }
            workgroupBarrier();
        }

        // Reconstruct path from best_seg.
        if lidx == 0u && is_valid_req {
            let safe_person_id = min(req.person_id, 524287u);
            let base_path = safe_person_id * params.max_path_len;

            var cur = best_seg;
            var path_tmp: array<u32, 512>;
            var count: u32 = 0u;

            while count < params.max_path_len {
                path_tmp[count] = cur;
                count = count + 1u;
                let p = get_prev_ht(base_prev, cur);
                if p < 0 || u32(p) == cur { break; }
                cur = u32(p);
            }

            for (var k: u32 = 0u; k < count; k = k + 1u) {
                paths[base_path + k] = path_tmp[count - 1u - k];
            }
            for (var k: u32 = count; k < params.max_path_len; k = k + 1u) {
                paths[base_path + k] = 0xFFFFFFFFu;
            }
        }
    }
}
