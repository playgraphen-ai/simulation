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
    b_start: u32,
    b_count: u32,
    logic_start: u32,
    logic_count: u32,
    r_start: u32,
    r_count: u32,
    cycle_frames: u32,
    do_readback: u32,
    reset_stats: u32,
};

struct PathRequest {
    start: u32,
    target_seg: u32,
    person_id: u32,
};

struct PathRequestQueue {
    count_x: atomic<u32>,
    count_y: u32,
    count_z: u32,
    processed: atomic<u32>,
    requests: array<PathRequest>,
};

struct GpuStats {
    people_count: atomic<u32>,
    home_count: atomic<u32>,
    work_count: atomic<u32>,
    shop_count: atomic<u32>,
    travelling_count: atomic<u32>,
    total_money: atomic<u32>,
    residential_occupancy: atomic<u32>,
    office_occupancy: atomic<u32>,
    shop_occupancy: atomic<u32>,
    residential_count: atomic<u32>,
    office_count: atomic<u32>,
    shop_count_b: atomic<u32>,
};

@group(0) @binding(0) var people_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var roads_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(2) var buildings_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(4) var<storage, read_write> path_queue: PathRequestQueue;
@group(0) @binding(5) var<storage, read_write> person_paths: array<u32>;
@group(0) @binding(6) var<storage, read_write> congestion: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> stats: GpuStats;


fn person_coords(pid: u32) -> array<vec2<i32>, 3> {
    let base = i32(pid * 3u);
    let w = max(1, i32(params.people_tex_w));
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    let c2 = vec2<i32>((base + 2) % w, (base + 2) / w);
    return array<vec2<i32>, 3>(c0, c1, c2);
}

fn building_coords(bid: u32) -> array<vec2<i32>, 3> {
    let base = i32(bid * 3u);
    let w = max(1, i32(params.buildings_tex_w));
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    let c2 = vec2<i32>((base + 2) % w, (base + 2) / w);
    return array<vec2<i32>, 3>(c0, c1, c2);
}

fn road_coords(sid: u32) -> array<vec2<i32>, 2> {
    let base = i32(sid * 2u);
    let w = max(1, i32(params.roads_tex_w));
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    return array<vec2<i32>, 2>(c0, c1);
}

const ACT_TRAVEL: f32 = 0.0;
const ACT_HOME:   f32 = 1.0;
const ACT_WORK:   f32 = 2.0;
const ACT_SHOP:   f32 = 3.0;

fn rand(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return f32(x) / 4294967296.0;
}

@compute @workgroup_size(64)
fn main_people_movement(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.people_count { return; }

    let coords = person_coords(pid);
    var texel0 = textureLoad(people_tex, coords[0]);
    var texel1 = textureLoad(people_tex, coords[1]);
    var texel2 = textureLoad(people_tex, coords[2]);

    if texel0.x == 0.0 { return; } // Not spawned

    var destination = texel0.z;
    let home = texel0.w;
    var activity = texel1.y;
    var activity_time = texel1.z;
    var path_cursor = texel1.w;
    var current_seg = texel2.x;
    var prev_seg = texel2.y;

    if activity == ACT_TRAVEL {
        let base_idx = pid * 256u;
        let current_step = u32(path_cursor);
        
        let max_paths = 16777216u; // 65536 * 256
        if base_idx + current_step >= max_paths {
            return;
        }

        let current_path_seg = person_paths[base_idx + current_step];

        if current_path_seg != 0xFFFFFFFFu {
            if activity_time < 0.0 {
                // Just received path, start on first segment!
                let r_coords = road_coords(current_path_seg);
                let r_tex1 = textureLoad(roads_tex, r_coords[1]);
                activity_time = max(0.5, r_tex1.w); // length is tex1.w
                current_seg = f32(current_path_seg);
                prev_seg = current_seg; // No previous segment yet
            } else {
                let r_coords = road_coords(current_path_seg);
                let r_tex1 = textureLoad(roads_tex, r_coords[1]);
                let speed = max(0.05, r_tex1.x); // speed_mean is tex1.x
                
                // Write to congestion buffer: Simple increment instead of bitmask
                atomicAdd(&congestion[current_path_seg], 1u);

                activity_time = activity_time - speed * params.dt;

                if activity_time <= 0.0 {
                    path_cursor = path_cursor + 1.0;
                    let next_step = u32(path_cursor);
                    
                    if next_step < 256u && base_idx + next_step < max_paths {
                        let next_path_seg = person_paths[base_idx + next_step];
                        if next_path_seg != 0xFFFFFFFFu {
                            let nr_coords = road_coords(next_path_seg);
                            let nr_tex1 = textureLoad(roads_tex, nr_coords[1]);
                            activity_time = max(0.5, nr_tex1.w); // length is tex1.w
                            prev_seg = current_seg;
                            current_seg = f32(next_path_seg);
                        } else {
                            // Arrived!
                            let dest_b_coords = building_coords(u32(destination));
                            let b_tex0 = textureLoad(buildings_tex, dest_b_coords[0]);
                            var b_tex1 = textureLoad(buildings_tex, dest_b_coords[1]);

                            let btype = b_tex0.z;
                            if btype == 0.0 { activity = ACT_HOME; activity_time = params.home_duration; }
                            else if btype == 1.0 { activity = ACT_WORK; activity_time = params.work_duration; }
                            else if btype == 2.0 { activity = ACT_SHOP; activity_time = params.shop_duration; }
                            else { activity = ACT_HOME; activity_time = params.home_duration; }

                            b_tex1.y = b_tex1.y + 1.0;
                            textureStore(buildings_tex, dest_b_coords[1], b_tex1);
                            path_cursor = 0.0;
                        }
                    } else {
                        // Max path len reached without arriving? Fallback.
                        activity = ACT_HOME;
                        activity_time = params.home_duration;
                        destination = home;
                    }
                }
            }
        } else {
            // No path found yet or failed. Timeout logic.
            activity_time = activity_time + params.dt;
            if activity_time > -1.0 {
                // About 9 seconds elapsed without finding a path. Let's force them home!
                activity = ACT_HOME;
                activity_time = params.home_duration;
                destination = home;
                
                let h_coords = building_coords(u32(home));
                var h_tex1 = textureLoad(buildings_tex, h_coords[1]);
                h_tex1.y = h_tex1.y + 1.0;
                textureStore(buildings_tex, h_coords[1], h_tex1);
            }
        }
    } else if activity == ACT_HOME || activity == ACT_WORK || activity == ACT_SHOP {
        activity_time = activity_time - params.dt;
    }

    texel0.z = destination;
    texel1.y = activity;
    texel1.z = activity_time;
    texel1.w = path_cursor;
    texel2.x = current_seg;
    texel2.y = prev_seg;

    textureStore(people_tex, coords[0], texel0);
    textureStore(people_tex, coords[1], texel1);
    textureStore(people_tex, coords[2], texel2);
}

@compute @workgroup_size(64)
fn main_people_logic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x + params.logic_start;
    if pid >= params.logic_start + params.logic_count || pid >= params.people_count { return; }

    let coords = person_coords(pid);
    var texel0 = textureLoad(people_tex, coords[0]);
    var texel1 = textureLoad(people_tex, coords[1]);
    var texel2 = textureLoad(people_tex, coords[2]);

    // Statistics
    atomicAdd(&stats.people_count, 1u);
    let current_activity = texel1.y;
    if (current_activity == ACT_HOME) { atomicAdd(&stats.home_count, 1u); }
    else if (current_activity == ACT_WORK) { atomicAdd(&stats.work_count, 1u); }
    else if (current_activity == ACT_SHOP) { atomicAdd(&stats.shop_count, 1u); }
    else if (current_activity == ACT_TRAVEL) { atomicAdd(&stats.travelling_count, 1u); }
    atomicAdd(&stats.total_money, u32(max(0.0, texel0.x)));

    var money = texel0.x;
    var rng_state = params.rng_seed + pid + 777u;

    if money == 0.0 && params.buildings_count > 0u {
        var home_id = 0u;
        var found_home = false;
        for (var i = 0u; i < 20u; i = i + 1u) {
            let bid = u32(rand(&rng_state) * f32(params.buildings_count));
            let b_coords = building_coords(bid);
            let b_tex0 = textureLoad(buildings_tex, b_coords[0]);
            var b_tex1 = textureLoad(buildings_tex, b_coords[1]);
            if b_tex0.z == 0.0 && b_tex1.y < b_tex1.z { // Residential and has space
                home_id = bid;
                found_home = true;
                b_tex1.y = b_tex1.y + 1.0;
                textureStore(buildings_tex, b_coords[1], b_tex1);
                break;
            }
        }
        
        if found_home {
            var work_id = home_id;
            for (var i = 0u; i < 20u; i = i + 1u) {
                let bid = u32(rand(&rng_state) * f32(params.buildings_count));
                let b_coords = building_coords(bid);
                let b_tex0 = textureLoad(buildings_tex, b_coords[0]);
                let b_tex1 = textureLoad(buildings_tex, b_coords[1]);
                if b_tex0.z == 1.0 && b_tex1.y < b_tex1.z { // Office and has space
                    work_id = bid;
                    break;
                }
            }

            let home_coords = building_coords(home_id);
            let h_tex1 = textureLoad(buildings_tex, home_coords[1]);
            let home_seg = h_tex1.w;
            let target_b_coords = building_coords(work_id);
            let target_b_tex1 = textureLoad(buildings_tex, target_b_coords[1]);
            let target_seg = u32(target_b_tex1.w);

            texel0 = vec4<f32>(50.0 + rand(&rng_state) * 450.0, 18.0 + rand(&rng_state) * 57.0, f32(work_id), f32(home_id));
            texel1 = vec4<f32>(f32(work_id), 0.0, -10.0, 0.0); // Travel, waiting for path
            texel2 = vec4<f32>(home_seg, home_seg, 0.0, 0.0);
            
            // Re-load variables for simulation
            money = texel0.x;

            // --- QUEUE INITIAL PATH REQUEST ---
            let req_idx = atomicAdd(&path_queue.count_x, 1u);
            let max_queue = 16384u;
            if req_idx < max_queue {
                path_queue.requests[req_idx] = PathRequest(u32(home_seg), target_seg, pid);
            }
            person_paths[pid * 256u] = 0xFFFFFFFFu;
            
            textureStore(people_tex, coords[0], texel0);
            textureStore(people_tex, coords[1], texel1);
            textureStore(people_tex, coords[2], texel2);
        }
        return;
    } else if money == 0.0 {
        return;
    }

    var destination = texel0.z;
    let home = texel0.w;
    let work = texel1.x;
    var activity = texel1.y;
    var activity_time = texel1.z;
    var path_cursor = texel1.w;

    if activity == ACT_HOME || activity == ACT_WORK || activity == ACT_SHOP {
        if activity_time <= 0.0 {
            // Activity done — pick next destination and apply economy.
            var current_building: u32 = 0u;
            if activity == ACT_HOME {
                money = max(0.0, money - params.rent_cost);
                current_building = u32(home);
            } else if activity == ACT_WORK {
                money = money + params.work_salary;
                current_building = u32(work);
            } else if activity == ACT_SHOP {
                money = max(0.0, money - params.shop_cost);
                current_building = u32(destination);
            }

            // Release building occupancy
            let current_b_coords = building_coords(current_building);
            var current_b_tex1 = textureLoad(buildings_tex, current_b_coords[1]);
            current_b_tex1.y = max(0.0, current_b_tex1.y - 1.0);
            textureStore(buildings_tex, current_b_coords[1], current_b_tex1);
            let start_seg = u32(current_b_tex1.w); // road_seg is tex1.w

            // Pick next activity
            var next_activity = ACT_HOME;
            if activity == ACT_HOME {
                if rand(&rng_state) < params.home_to_work_prob {
                    next_activity = ACT_WORK;
                } else {
                    next_activity = ACT_SHOP;
                }
            }

            // Pick next destination
            var next_dest = home;
            if next_activity == ACT_WORK {
                next_dest = work;
            } else if next_activity == ACT_SHOP {
                var found_shop = false;
                for (var probes = 0u; probes < 20u; probes = probes + 1u) {
                    let candidate = u32(rand(&rng_state) * f32(params.buildings_count));
                    let b_coords = building_coords(candidate);
                    let b_tex0 = textureLoad(buildings_tex, b_coords[0]);
                    let b_tex1 = textureLoad(buildings_tex, b_coords[1]);
                    let btype = b_tex0.z;
                    let capacity = b_tex1.z;
                    if btype == 2.0 && capacity > 0.0 {
                        next_dest = f32(candidate);
                        found_shop = true;
                        break;
                    }
                }
                if !found_shop {
                    next_activity = ACT_HOME; // fallback
                    next_dest = home;
                }
            }

            let next_b_coords = building_coords(u32(next_dest));
            let next_b_tex1 = textureLoad(buildings_tex, next_b_coords[1]);
            let target_seg = u32(next_b_tex1.w);

            // Queue path request
            let req_idx = atomicAdd(&path_queue.count_x, 1u);
            let max_queue = 16384u;
            if req_idx < max_queue {
                path_queue.requests[req_idx] = PathRequest(start_seg, target_seg, pid);
            }

            // Reset path buffer for this person
            let path_idx = pid * 256u;
            if path_idx < 16777216u { // 65536 * 256 = 16777216
                person_paths[path_idx] = 0xFFFFFFFFu;
            }

            destination = next_dest;
            activity = ACT_TRAVEL;
            activity_time = -10.0; // Negative means waiting for path (timeout timer)
            path_cursor = 0.0;
            
            texel0.x = money;
            texel0.z = destination;
            texel1.y = activity;
            texel1.z = activity_time;
            texel1.w = path_cursor;
            
            textureStore(people_tex, coords[0], texel0);
            textureStore(people_tex, coords[1], texel1);
        }
    }
}

@compute @workgroup_size(64)
fn main_buildings(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bid = gid.x + params.b_start;
    if bid >= params.b_start + params.b_count || bid >= params.buildings_count { return; }

    let coords = building_coords(bid);
    var tex0 = textureLoad(buildings_tex, coords[0]);
    var tex1 = textureLoad(buildings_tex, coords[1]);
    var tex2 = textureLoad(buildings_tex, coords[2]);

    let btype = tex0.z;
    let occupancy = tex1.y;
    if (btype == 0.0) {
        atomicAdd(&stats.residential_count, 1u);
        atomicAdd(&stats.residential_occupancy, u32(occupancy));
    } else if (btype == 1.0) {
        atomicAdd(&stats.office_count, 1u);
        atomicAdd(&stats.office_occupancy, u32(occupancy));
    } else if (btype == 2.0) {
        atomicAdd(&stats.shop_count_b, 1u);
        atomicAdd(&stats.shop_occupancy, u32(occupancy));
    }

    var level = tex0.w;
    var occupants = tex1.y;
    var capacity = tex1.z;

    var growth = tex2.x;
    var age_seconds = tex2.y;

    if capacity == 0.0 { return; }

    let logic_dt = params.dt * f32(params.cycle_frames);
    age_seconds = age_seconds + logic_dt;

    let fill = occupants / max(1.0, capacity);

    if fill > 0.7 && level < 4.0 {
        growth = growth + (0.1 * logic_dt);
        if growth >= 1.0 {
            level = level + 1.0;
            growth = 0.0;
            let base_cap = select(select(4.0, 6.0, btype == 1.0), 8.0, btype == 2.0);
            capacity = base_cap * exp2(level);
            let base_inc = select(select(2.0, 5.0, btype == 1.0), 3.0, btype == 2.0);
            tex1.x = base_inc * (level + 1.0);
            tex1.z = capacity;
        }
    } else if fill < 0.05 && age_seconds > 60.0 {
        growth = growth - (0.02 * params.abandon_multiplier * logic_dt);
        if growth <= -1.0 {
            capacity = 0.0;
            tex1.z = 0.0;
            occupants = 0.0;
        }
    } else {
        growth = min(0.0, max(-1.0, growth + (0.01 * logic_dt)));
    }

    tex0.w = level;
    tex1.y = occupants;
    tex2.x = growth;
    tex2.y = age_seconds;

    textureStore(buildings_tex, coords[0], tex0);
    textureStore(buildings_tex, coords[1], tex1);
    textureStore(buildings_tex, coords[2], tex2);
}
