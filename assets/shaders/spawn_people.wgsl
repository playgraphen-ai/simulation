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
    tax_income: f32,
    tax_rent: f32,
    tax_consumption: f32,
    buildings_tex_w: u32,
    buildings_tex_h: u32,
    roads_tex_w: u32,
    buildings_count: u32,
    segments_count: u32,
    spawn_count: u32,
    spawn_start_index: u32,
    b_start: u32,
    b_count: u32,
    logic_start: u32,
    logic_count: u32,
    r_start: u32,
    r_count: u32,
    cycle_frames: u32,
    do_readback: u32,
    reset_stats: u32,
    grid_w: u32,
    grid_h: u32,
    entry_seg: u32,
    collisions_enabled: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
@group(0) @binding(9) var<storage, read_write> building_stats: array<atomic<u32>>; // occupants: idx*3, assigned: idx*3+1, tax: idx*3+2

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

fn road_coords(sid: u32) -> array<vec2<i32>, 5> {
    let base = i32(sid * 5u);
    let w = max(1, i32(params.roads_tex_w));
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    let c2 = vec2<i32>((base + 2) % w, (base + 2) / w);
    let c3 = vec2<i32>((base + 3) % w, (base + 3) / w);
    let c4 = vec2<i32>((base + 4) % w, (base + 4) / w);
    return array<vec2<i32>, 5>(c0, c1, c2, c3, c4);
}

fn pcg_hash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return f32(x) / 4294967296.0;
}

const ACT_TRAVEL: f32 = 0.0;
const ACT_HOME:   f32 = 1.0;
const ACT_WORK:   f32 = 2.0;
const ACT_SHOP:   f32 = 3.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let request_idx = gid.x;
    if request_idx >= params.spawn_count { return; }

    let pid = params.spawn_start_index + request_idx;
    var rng_state = pcg_hash(params.rng_seed + pid + 777u);

    // 1. Find Home (Residential)
    var home_id = 0u;
    var found_home = false;
    for (var i = 0u; i < 50u; i = i + 1u) {
        let bid = u32(rand(&rng_state) * f32(params.buildings_count));
        let coords = building_coords(bid);
        let tex0 = textureLoad(buildings_tex, coords[0]);
        var tex1 = textureLoad(buildings_tex, coords[1]);
        
        let btype = tex0.z; // 0=Res, 1=Off, 2=Shop
        let capacity = tex1.z;
        let assigned = atomicLoad(&building_stats[bid * 3u + 1u]);
        
        if btype == 0.0 && f32(assigned) < capacity {
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
                    
                    atomicAdd(&building_stats[bid * 3u + 1u], 1u);
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
        let capacity = tex1.z;
        let assigned = atomicLoad(&building_stats[bid * 3u + 1u]);
        
        if btype == 1.0 && f32(assigned) < capacity {
            work_id = bid;
            found_work = true;
            atomicAdd(&building_stats[bid * 3u + 1u], 1u);
            break;
        }
    }

    // 3. Roll for initial activity
    let total_time = params.home_duration + (params.home_to_work_prob * params.work_duration) + ((1.0 - params.home_to_work_prob) * params.shop_duration);
    let roll = rand(&rng_state) * total_time;
    
    var activity = ACT_HOME;
    var activity_time = rand(&rng_state) * params.home_duration;
    var current_bid = home_id;
    
    if roll > params.home_duration {
        if roll < params.home_duration + (params.home_to_work_prob * params.work_duration) {
            activity = ACT_WORK;
            activity_time = rand(&rng_state) * params.work_duration;
            current_bid = work_id;
        } else {
            // Find a Shop
            var found_shop = false;
            for (var i = 0u; i < 50u; i = i + 1u) {
                let bid = u32(rand(&rng_state) * f32(params.buildings_count));
                let coords = building_coords(bid);
                let tex0 = textureLoad(buildings_tex, coords[0]);
                let tex1 = textureLoad(buildings_tex, coords[1]);
                if tex0.z == 2.0 && tex1.z > 0.0 {
                    current_bid = bid;
                    found_shop = true;
                    break;
                }
            }
            if found_shop {
                activity = ACT_SHOP;
                activity_time = rand(&rng_state) * params.shop_duration;
            } else {
                // Fallback to home
                activity = ACT_HOME;
                activity_time = rand(&rng_state) * params.home_duration;
                current_bid = home_id;
            }
        }
    }

    // 4. Initialize Person
    let p_coords = person_coords(pid);
    
    let curr_coords = building_coords(current_bid);
    let c_tex1 = textureLoad(buildings_tex, curr_coords[1]);
    let c_tex2 = textureLoad(buildings_tex, curr_coords[2]);
    let start_seg = c_tex1.w;
    let start_t = c_tex2.z;

    let target_b_coords = building_coords(u32(home_id)); // Default next destination
    var target_seg = textureLoad(buildings_tex, target_b_coords[1]).w;
    var target_t = textureLoad(buildings_tex, target_b_coords[2]).z;
    
    // If we are at home, next target depends on probability
    if activity == ACT_HOME {
        var next_dest = f32(work_id);
        if rand(&rng_state) > params.home_to_work_prob {
            // We don't have a shop assigned, so it will be picked when leaving home
            // For now, target work or stay home.
        }
        let next_b_coords = building_coords(u32(next_dest));
        target_seg = textureLoad(buildings_tex, next_b_coords[1]).w;
        target_t = textureLoad(buildings_tex, next_b_coords[2]).z;
    }

    let money = 50.0 + rand(&rng_state) * 450.0;

    let texel0 = vec4<f32>(money, -1.0, f32(current_bid), f32(home_id));
    let texel1 = vec4<f32>(f32(work_id), activity, activity_time, 0.0);
    let texel2 = vec4<f32>(f32(start_seg), f32(start_seg), start_t, target_t);

    textureStore(people_tex, p_coords[0], texel0);
    textureStore(people_tex, p_coords[1], texel1);
    textureStore(people_tex, p_coords[2], texel2);

    // Reset path buffer for this person
    let path_idx = pid * 512u;
    if path_idx < 268435456u { // 131072 * 512
        person_paths[path_idx] = 0xFFFFFFFFu;
    }
}
