// Compute shader: parallel spawn of people.
//
// Finds a home (Residential) and work (Office) for each new person.
// Uses probabilistic search with retries to find buildings with capacity.

struct SimParams {
    dt: f32,
    home_duration: f32,
    work_duration: f32,
    shop_duration: f32,
    home_to_work_prob: f32,
    people_count: u32,
    people_tex_w: u32,
    people_tex_h: u32,
    rng_seed: u32,
    abandon_multiplier: f32,
    rent_cost: f32,
    work_salary: f32,
    shop_cost: f32,
    buildings_tex_w: u32,
    buildings_tex_h: u32,
    roads_tex_w: u32,
    buildings_count: u32,
    segments_count: u32,
    spawn_count: u32,
    spawn_start_index: u32,
    grid_w: u32,
    grid_h: u32,
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

@group(0) @binding(0) var people_tex    : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var roads_tex     : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(2) var buildings_tex : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(4) var<storage, read_write> path_queue: PathRequestQueue;
@group(0) @binding(5) var<storage, read_write> person_paths: array<u32>;
// Binding 6: Congestion buffer
// Binding 7: Stats buffer
@group(0) @binding(8) var<storage, read_write> occupancy: array<atomic<u32>>;

fn person_coords(pid: u32) -> array<vec2<i32>, 3> {
    let base = i32(pid * 3u);
    let w = i32(params.people_tex_w);
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    let c2 = vec2<i32>((base + 2) % w, (base + 2) / w);
    return array<vec2<i32>, 3>(c0, c1, c2);
}

fn building_coords(bid: u32) -> array<vec2<i32>, 3> {
    let base = i32(bid * 3u);
    let w = i32(params.buildings_tex_w);
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    let c2 = vec2<i32>((base + 2) % w, (base + 2) / w);
    return array<vec2<i32>, 3>(c0, c1, c2);
}

fn road_coords(sid: u32) -> array<vec2<i32>, 4> {
    let base = i32(sid * 4u);
    let w = max(1, i32(params.roads_tex_w));
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    let c2 = vec2<i32>((base + 2) % w, (base + 2) / w);
    let c3 = vec2<i32>((base + 3) % w, (base + 3) / w);
    return array<vec2<i32>, 4>(c0, c1, c2, c3);
}

fn rand(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return f32(x) / 4294967296.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let request_idx = gid.x;
    if request_idx >= params.spawn_count { return; }

    let pid = params.spawn_start_index + request_idx;
    var rng_state = params.rng_seed + pid + 777u;

    // 1. Find Home (Residential)
    var home_id = 0u;
    var found_home = false;
    for (var i = 0u; i < 50u; i = i + 1u) {
        let bid = u32(rand(&rng_state) * f32(params.buildings_count));
        let coords = building_coords(bid);
        let tex0 = textureLoad(buildings_tex, coords[0]);
        var tex1 = textureLoad(buildings_tex, coords[1]);
        
        let btype = tex0.z; // 0=Res, 1=Off, 2=Shop
        let occupants = tex1.y;
        let capacity = tex1.z;
        
        if btype == 0.0 && occupants < capacity {
            let home_seg = u32(tex1.w);
            let r_coords = road_coords(home_seg);
            let r_tex0 = textureLoad(roads_tex, r_coords[0]);
            let r_tex1 = textureLoad(roads_tex, r_coords[1]);
            if r_tex1.x > 0.15 {
                // Check occupancy to ensure the spawn point isn't blocked
                let h_tex2 = textureLoad(buildings_tex, coords[2]);
                let home_t = h_tex2.z;
                let ax = r_tex0.x; let ay = r_tex0.y;
                let bx = r_tex0.z; let by = r_tex0.w;
                let tx = u32(mix(ax, bx, home_t));
                let ty = u32(mix(ay, by, home_t));
                let grid_idx = tx + ty * params.grid_w;
                
                var is_blocked = false;
                if grid_idx < params.grid_w * params.grid_h {
                    if atomicLoad(&occupancy[grid_idx]) > 0u {
                        is_blocked = true;
                    }
                }

                if !is_blocked {
                    home_id = bid;
                    found_home = true;
                    
                    // Increment occupants (probabilistic, non-atomic for now to avoid complexity, 
                    // but retries help. For thousands of spawns, some overlap is okay 
                    // as buildings will auto-correct next frame)
                    tex1.y = tex1.y + 1.0;
                    textureStore(buildings_tex, coords[1], tex1);
                    break;
                }
            }
        }
    }
    
    if !found_home { return; } // Failed to find home, slot stays dead

    // 2. Find Work (Office)
    var work_id = home_id;
    var found_work = false;
    for (var i = 0u; i < 50u; i = i + 1u) {
        let bid = u32(rand(&rng_state) * f32(params.buildings_count));
        let coords = building_coords(bid);
        let tex0 = textureLoad(buildings_tex, coords[0]);
        let tex1 = textureLoad(buildings_tex, coords[1]);
        
        let btype = tex0.z;
        let occupants = tex1.y;
        let capacity = tex1.z;
        
        if btype == 1.0 && occupants < capacity {
            work_id = bid;
            found_work = true;
            break;
        }
    }

    // 3. Initialize Person
    let p_coords = person_coords(pid);
    let home_coords = building_coords(home_id);
    let h_tex1 = textureLoad(buildings_tex, home_coords[1]);
    let h_tex2 = textureLoad(buildings_tex, home_coords[2]);
    let home_seg = h_tex1.w;
    let home_t = h_tex2.z;

    let work_coords = building_coords(work_id);
    let w_tex1 = textureLoad(buildings_tex, work_coords[1]);
    let w_tex2 = textureLoad(buildings_tex, work_coords[2]);
    let target_seg = w_tex1.w;
    let target_t = w_tex2.z;

    let money = 50.0 + rand(&rng_state) * 450.0;
    let age = 18.0 + rand(&rng_state) * 57.0;

    // activity_code: 0.0 = Travel (ACT_TRAVEL)
    // activity_time: -10.0 = Waiting for path timeout
    let texel0 = vec4<f32>(money, age, f32(work_id), f32(home_id));
    let texel1 = vec4<f32>(f32(work_id), 0.0, -10.0, 0.0);
    let texel2 = vec4<f32>(home_seg, home_seg, home_t, target_t);

    textureStore(people_tex, p_coords[0], texel0);
    textureStore(people_tex, p_coords[1], texel1);
    textureStore(people_tex, p_coords[2], texel2);

    // Queue path request
    let req_idx = atomicAdd(&path_queue.count_x, 1u);
    let max_queue = 16384u;
    if req_idx < max_queue {
        path_queue.requests[req_idx] = PathRequest(u32(home_seg), u32(target_seg), pid, 0u);
    }

    // Reset path buffer for this person
    let path_idx = pid * 256u;
    if path_idx < 16777216u {
        person_paths[path_idx] = 0xFFFFFFFFu;
    }
}
