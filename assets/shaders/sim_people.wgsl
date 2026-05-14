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
    do_stats_readback: u32,
    reset_stats: u32,
    grid_w: u32,
    grid_h: u32,
    entry_seg: u32,
    collisions_enabled: f32,
    recount_slice: u32,
    recount_slice_count: u32,
    _pad0: u32,
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
    residential_assigned: atomic<u32>,
    office_assigned: atomic<u32>,
    bankrupt_count: atomic<u32>,
    tax_income_total: atomic<u32>,
    tax_rent_total: atomic<u32>,
    tax_consumption_total: atomic<u32>,
    _pad: u32,
};

@group(0) @binding(0) var people_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var roads_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(2) var buildings_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(4) var<storage, read_write> path_queue: PathRequestQueue;
@group(0) @binding(5) var<storage, read_write> person_paths: array<u32>;
@group(0) @binding(6) var<storage, read_write> congestion: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> stats: GpuStats;
@group(0) @binding(8) var<storage, read_write> occupancy: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> building_stats: array<atomic<u32>>; // occupants: idx*3, assigned: idx*3+1, tax: idx*3+2

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

const ACT_TRAVEL: f32 = 0.0;
const ACT_HOME:   f32 = 1.0;
const ACT_WORK:   f32 = 2.0;
const ACT_SHOP:   f32 = 3.0;
const ACT_ARRIVED: f32 = 4.0;

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

@compute @workgroup_size(64)
fn main_people_occupancy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.people_count { return; }

    let coords = person_coords(pid);
    let tex1 = textureLoad(people_tex, coords[1]);
    let tex2 = textureLoad(people_tex, coords[2]);

    let activity = tex1.y;
    let current_seg = u32(tex2.x);
    let prev_seg = tex2.y;
    let activity_time = tex1.z;

    if activity == ACT_TRAVEL && current_seg != 0xFFFFFFFFu {
        let r_coords = road_coords(current_seg);
        let r_tex0 = textureLoad(roads_tex, r_coords[0]);
        let r_tex1 = textureLoad(roads_tex, r_coords[1]);
        let r_tex4 = textureLoad(roads_tex, r_coords[4]);
        let rtype = r_tex4.x;
        let seg_len = max(0.5, r_tex1.w);

        var start_at_b = false;
        if prev_seg >= 1000000.0 { start_at_b = true; }

        var frac = 0.0;
        if activity_time < 0.0 {
            frac = tex2.z; // start_t
        } else {
            let rem = clamp(activity_time / seg_len, 0.0, 1.0);
            if start_at_b { frac = rem; } else { frac = 1.0 - rem; }
        }

        // Determine tile with right-hand offset
        let ax = r_tex0.x; let ay = r_tex0.y;
        let bx = r_tex0.z; let by = r_tex0.w;
        
        let dir = normalize(vec2<f32>(bx - ax, by - ay));
        let side = vec2<f32>(-dir.y, dir.x);
        
        var offset = select(0.35, -0.35, start_at_b);
        if rtype == 1.0 {
            let lane = f32(pid % 2u);
            let lane_offset = 0.5 + lane * 1.0;
            offset = select(lane_offset, -lane_offset, start_at_b);
        } else if rtype == 2.0 {
            let lane = f32(pid % 4u);
            let lane_offset = 0.5 + lane * 0.675;
            offset = select(lane_offset, -lane_offset, start_at_b);
        }
        
        let pos = mix(vec2<f32>(ax, ay), vec2<f32>(bx, by), frac) + side * offset;
        let tx = u32(pos.x + 0.5);
        let ty = u32(pos.y + 0.5);
        
        let idx = tx + ty * params.grid_w;
        if idx < params.grid_w * params.grid_h {
            atomicAdd(&occupancy[idx], 1u);
        }
    }
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
        let base_idx = pid * 512u;
        let current_step = u32(path_cursor);
        
        let max_paths = 67108864u; // 131072 * 512
        if base_idx + current_step >= max_paths {
            return;
        }

        let current_path_seg = person_paths[base_idx + current_step];

        if current_path_seg != 0xFFFFFFFFu {
            let r_coords = road_coords(current_path_seg);
            let r_tex0 = textureLoad(roads_tex, r_coords[0]);
            let r_tex1 = textureLoad(roads_tex, r_coords[1]);
            let seg_len = max(0.5, r_tex1.w);

            let real_prev_seg = u32(select(prev_seg, prev_seg - 1000000.0, prev_seg >= 1000000.0));

            // Determine if we are travelling from A to B or B to A
            var start_at_b = false;
            var is_jump = false; // Flag for hierarchical non-contiguous path

            if real_prev_seg != u32(current_seg) && real_prev_seg != 0xFFFFFFFFu {
                let p_coords = road_coords(real_prev_seg);
                let p_tex0 = textureLoad(roads_tex, p_coords[0]);
                let seg_b = vec2<f32>(r_tex0.z, r_tex0.w);
                let seg_a = vec2<f32>(r_tex0.x, r_tex0.y);
                
                // Normal adjacency
                if (p_tex0.x == seg_b.x && p_tex0.y == seg_b.y) || (p_tex0.z == seg_b.x && p_tex0.w == seg_b.y) {
                    start_at_b = true;
                } else if !((p_tex0.x == seg_a.x && p_tex0.y == seg_a.y) || (p_tex0.z == seg_a.x && p_tex0.w == seg_a.y)) {
                    // Not connected at all -> Hierarchical jump!
                    // We just arrived at a highway from a local road, or vice versa.
                    is_jump = true;
                    // Pick an arbitrary start direction (e.g. A to B)
                    start_at_b = false;
                }
            } else if real_prev_seg == u32(current_seg) {
                let next_step = current_step + 1u;
                let next_path_seg = person_paths[base_idx + next_step];
                if next_path_seg != 0xFFFFFFFFu {
                    let nr_coords = road_coords(next_path_seg);
                    let nr_tex0 = textureLoad(roads_tex, nr_coords[0]);
                    let seg_a = vec2<f32>(r_tex0.x, r_tex0.y);
                    let seg_b = vec2<f32>(r_tex0.z, r_tex0.w);
                    if (nr_tex0.x == seg_a.x && nr_tex0.y == seg_a.y) || (nr_tex0.z == seg_a.x && nr_tex0.w == seg_a.y) {
                        start_at_b = true;
                    } else if !((nr_tex0.x == seg_b.x && nr_tex0.y == seg_b.y) || (nr_tex0.z == seg_b.x && nr_tex0.w == seg_b.y)) {
                        is_jump = true;
                        start_at_b = false; // Default
                    }
                } else {
                    if texel2.z > texel2.w { start_at_b = true; }
                }
            }

            if activity_time < 0.0 || is_jump {
                let start_t = select(texel2.z, 0.0, is_jump); // If jump, start at beginning of segment
                if start_at_b {
                    activity_time = seg_len * start_t;
                    prev_seg = f32(current_path_seg) + 1000000.0;
                } else {
                    activity_time = seg_len * (1.0 - start_t);
                    prev_seg = f32(current_path_seg);
                }
                current_seg = f32(current_path_seg);
            } else {
                let speed = max(0.05, r_tex1.x); 
                
                // --- Collision Avoidance ---
                var current_frac = 0.0;
                let rem = clamp(activity_time / seg_len, 0.0, 1.0);
                if start_at_b { current_frac = rem; } else { current_frac = 1.0 - rem; }

                // Look ahead 1.5 units
                let look_ahead = 1.5 / seg_len;
                var ahead_frac = current_frac + select(look_ahead, -look_ahead, start_at_b);
                ahead_frac = clamp(ahead_frac, 0.0, 1.0);

                let ax = r_tex0.x; let ay = r_tex0.y;
                let bx = r_tex0.z; let by = r_tex0.w;
                
                let r_tex4 = textureLoad(roads_tex, r_coords[4]);
                let rtype = r_tex4.x;

                let dir = normalize(vec2<f32>(bx - ax, by - ay));
                let side = vec2<f32>(-dir.y, dir.x);
                
                var offset = select(0.35, -0.35, start_at_b);
                if rtype == 1.0 {
                    let lane = f32(pid % 2u);
                    let lane_offset = 0.5 + lane * 1.0;
                    offset = select(lane_offset, -lane_offset, start_at_b);
                } else if rtype == 2.0 {
                    let lane = f32(pid % 4u);
                    let lane_offset = 0.5 + lane * 0.675;
                    offset = select(lane_offset, -lane_offset, start_at_b);
                }

                let current_pos = mix(vec2<f32>(ax, ay), vec2<f32>(bx, by), current_frac) + side * offset;
                var reserved_idx = i32(texel0.y); // Read persistent occupancy index

                let pos = mix(vec2<f32>(ax, ay), vec2<f32>(bx, by), ahead_frac) + side * offset;
                let tx = u32(pos.x + 0.5);
                let ty = u32(pos.y + 0.5);
                let ahead_idx = tx + ty * params.grid_w;

                var is_blocked = false;
                if params.collisions_enabled > 0.5 {
                    if ahead_idx < params.grid_w * params.grid_h {
                        let ahead_idx_i32 = i32(ahead_idx);
                        if ahead_idx_i32 != reserved_idx {
                            let old_occ = atomicExchange(&occupancy[ahead_idx], 1u);
                            if old_occ > 0u {
                                is_blocked = true;
                            } else {
                                // We claimed it! Free old space.
                                if reserved_idx >= 0 && reserved_idx < i32(params.grid_w * params.grid_h) {
                                    atomicStore(&occupancy[reserved_idx], 0u);
                                }
                                reserved_idx = ahead_idx_i32;
                                texel0.y = f32(reserved_idx); // Update persistent storage
                            }
                        }
                    }

                    // If at the end of segment, check next segment (peeking only, no lock yet)
                    if !is_blocked && ((start_at_b && ahead_frac <= 0.01) || (!start_at_b && ahead_frac >= 0.99)) {
                        let next_step = current_step + 1u;
                        let next_path_seg = person_paths[base_idx + next_step];
                        if next_path_seg != 0xFFFFFFFFu {
                            let nr_coords = road_coords(next_path_seg);
                            let n_tex0 = textureLoad(roads_tex, nr_coords[0]);
                            let n_tex4 = textureLoad(roads_tex, nr_coords[4]);
                            let n_ax = n_tex0.x; let n_ay = n_tex0.y;
                            let n_bx = n_tex0.z; let n_by = n_tex0.w;
                            let n_rtype = n_tex4.x;
                            
                            var next_start_at_b = false;
                            let nseg_b = vec2<f32>(n_tex0.z, n_tex0.w);
                            if (r_tex0.x == nseg_b.x && r_tex0.y == nseg_b.y) || (r_tex0.z == nseg_b.x && r_tex0.w == nseg_b.y) {
                                next_start_at_b = true;
                            }
                            
                            let n_dir = normalize(vec2<f32>(n_bx - n_ax, n_by - n_ay));
                            let n_side = vec2<f32>(-n_dir.y, n_dir.x);
                            
                            var n_offset = select(0.35, -0.35, next_start_at_b);
                            if n_rtype == 1.0 {
                                let lane = f32(pid % 2u);
                                let lane_offset = 0.5 + lane * 1.0;
                                n_offset = select(lane_offset, -lane_offset, next_start_at_b);
                            } else if n_rtype == 2.0 {
                                let lane = f32(pid % 4u);
                                let lane_offset = 0.5 + lane * 0.675;
                                n_offset = select(lane_offset, -lane_offset, next_start_at_b);
                            }

                            let n_frac = select(0.1, 0.9, next_start_at_b);
                            
                            let n_pos = mix(vec2<f32>(n_ax, n_ay), vec2<f32>(n_bx, n_by), n_frac) + n_side * n_offset;
                            let ntx = u32(n_pos.x + 0.5);
                            let nty = u32(n_pos.y + 0.5);
                            let n_idx = ntx + nty * params.grid_w;
                            
                            if n_idx < params.grid_w * params.grid_h {
                                let n_occ = atomicLoad(&occupancy[n_idx]);
                                if (i32(n_idx) != reserved_idx && n_occ > 0u) {
                                    is_blocked = true;
                                }
                            }
                        }
                    }
                }

                if !is_blocked {
                    activity_time = activity_time - speed * params.dt;
                    let safe_seg = min(current_path_seg, 65535u);
                    atomicAdd(&congestion[safe_seg], 1u);
                }

                // Determine stop threshold for last segment
                var stop_at = 0.0;
                let next_step = u32(path_cursor) + 1u;
                let next_path_seg = person_paths[base_idx + next_step];
                if next_path_seg == 0xFFFFFFFFu {
                    let target_t = texel2.w;
                    if start_at_b { stop_at = seg_len * target_t; } else { stop_at = seg_len * (1.0 - target_t); }
                }

                if activity_time <= stop_at {
                    let overshoot = activity_time - stop_at; 
                    path_cursor = path_cursor + 1.0;
                    
                    if next_path_seg != 0xFFFFFFFFu {
                        let nr_coords = road_coords(next_path_seg);
                        let nr_tex1 = textureLoad(roads_tex, nr_coords[1]);
                        let next_seg_len = max(0.5, nr_tex1.w);

                        var next_start_at_b = false;
                        var next_is_jump = false;
                        let n_tex0 = textureLoad(roads_tex, nr_coords[0]);
                        let nseg_b = vec2<f32>(n_tex0.z, n_tex0.w);
                        let nseg_a = vec2<f32>(n_tex0.x, n_tex0.y);
                        
                        if (r_tex0.x == nseg_b.x && r_tex0.y == nseg_b.y) || (r_tex0.z == nseg_b.x && r_tex0.w == nseg_b.y) {
                            next_start_at_b = true;
                        } else if !((r_tex0.x == nseg_a.x && r_tex0.y == nseg_a.y) || (r_tex0.z == nseg_a.x && r_tex0.w == nseg_a.y)) {
                            next_is_jump = true;
                        }

                        if next_is_jump {
                            // If jumping to the next segment, reset time completely (teleport)
                            activity_time = next_seg_len;
                            // Set a special prev_seg to force the jump logic on the next frame
                            prev_seg = f32(0xFFFFFFFFu);
                        } else {
                            activity_time = next_seg_len + overshoot;
                            if next_start_at_b {
                                prev_seg = f32(current_path_seg) + 1000000.0;
                            } else {
                                prev_seg = f32(current_path_seg);
                            }
                        }
                        current_seg = f32(next_path_seg);
                    } else {
                        // Arrived!
                        let dest_bid = u32(destination);
                        // atomicAdd(&building_stats[dest_bid * 3u], 1u); // Increment physical occupants

                        path_cursor = 0.0;
                        activity = ACT_ARRIVED;
                        activity_time = stop_at;
                        
                        // Release road occupancy reservation
                        if reserved_idx >= 0 && reserved_idx < i32(params.grid_w * params.grid_h) {
                            atomicStore(&occupancy[reserved_idx], 0u);
                            reserved_idx = -1;
                            texel0.y = -1.0;
                        }
                    }
                }
            }
        } else {
            // No path found yet or failed. Timeout logic.
            activity_time = activity_time + params.dt;
            if activity_time > -1.0 {
                activity = ACT_HOME;
                activity_time = params.home_duration;
                destination = home;
                let h_bid = u32(home);
                // atomicAdd(&building_stats[h_bid * 3u], 1u); // Increment physical occupants
            }
        }
    } else if activity == ACT_ARRIVED {
        let dest_b_coords = building_coords(u32(destination));
        let b_tex0 = textureLoad(buildings_tex, dest_b_coords[0]);
        let btype = b_tex0.z;
        
        if btype == 0.0 { activity = ACT_HOME; activity_time = params.home_duration; }
        else if btype == 1.0 { activity = ACT_WORK; activity_time = params.work_duration; }
        else if btype == 2.0 { activity = ACT_SHOP; activity_time = params.shop_duration; }
        else { activity = ACT_HOME; activity_time = params.home_duration; }
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
    // atomicAdd(&stats.people_count, 1u);
    let current_activity = texel1.y;
    // if (current_activity == ACT_HOME) { atomicAdd(&stats.home_count, 1u); }
    // else if (current_activity == ACT_WORK) { atomicAdd(&stats.work_count, 1u); }
    // else if (current_activity == ACT_SHOP) { atomicAdd(&stats.shop_count, 1u); }
    // else if (current_activity == ACT_TRAVEL) { atomicAdd(&stats.travelling_count, 1u); }
    // atomicAdd(&stats.total_money, u32(max(0.0, texel0.x)));

    var money = texel0.x;
    var rng_state = params.rng_seed + pid + 777u;

    if money == 0.0 && params.buildings_count > 0u {
        // --- RELEASE OLD BUILDINGS ---
        let old_home_f = texel0.w;
        let old_work_f = texel1.x;
        let reserved_idx_f = texel0.y;
        
        // Release road occupancy reservation
        if reserved_idx_f >= 0.0 {
            let reserved_idx = i32(reserved_idx_f);
            if reserved_idx < i32(params.grid_w * params.grid_h) {
                atomicStore(&occupancy[reserved_idx], 0u);
            }
            texel0.y = -1.0;
        }
        let old_home = u32(old_home_f);
        let old_work = u32(old_work_f);

        if old_home_f >= 0.0 {
            atomicSub(&building_stats[old_home * 3u + 1u], 1u); // Release assigned

            // Also release physical occupancy if they were there
            if current_activity == ACT_HOME || (current_activity == ACT_ARRIVED && u32(texel0.z) == old_home) {
                atomicSub(&building_stats[old_home * 3u], 1u);
            }
        }
        if old_work_f >= 0.0 && old_work != old_home {
            atomicSub(&building_stats[old_work * 3u + 1u], 1u); // Release assigned

            // Also release physical occupancy if they were there
            if current_activity == ACT_WORK || (current_activity == ACT_ARRIVED && u32(texel0.z) == old_work) {
                atomicSub(&building_stats[old_work * 3u], 1u);
            }
        }
        if current_activity == ACT_SHOP || (current_activity == ACT_ARRIVED && u32(texel0.z) != old_home && u32(texel0.z) != old_work) {
            let shop_id = u32(texel0.z);
            if texel0.z >= 0.0 {
                atomicSub(&building_stats[shop_id * 3u], 1u); // Release physical
            }
        }


        var home_id = 0u;
        var found_home = false;
        for (var i = 0u; i < 20u; i = i + 1u) {
            let bid = u32(rand(&rng_state) * f32(params.buildings_count));
            let b_coords = building_coords(bid);
            let b_tex0 = textureLoad(buildings_tex, b_coords[0]);
            let b_tex1 = textureLoad(buildings_tex, b_coords[1]);
            
            let assigned_at = atomicLoad(&building_stats[bid * 3u + 1u]);
            if b_tex0.z == 0.0 && f32(assigned_at) < b_tex1.z { // Residential and has space
                let home_seg = u32(b_tex1.w);
                let r_coords = road_coords(home_seg);
                let r_tex1 = textureLoad(roads_tex, r_coords[1]);
                if r_tex1.x <= 0.15 { continue; } // Road is full, try another or wait

                home_id = bid;
                found_home = true;
                atomicAdd(&building_stats[bid * 3u + 1u], 1u); // Increment assigned
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

                let assigned_at = atomicLoad(&building_stats[bid * 3u + 1u]);
                if b_tex0.z == 1.0 && f32(assigned_at) < b_tex1.z { // Office and has space
                    work_id = bid;
                    atomicAdd(&building_stats[bid * 3u + 1u], 1u); // Increment assigned
                    break;
                }
            }


            let home_coords = building_coords(home_id);
            let h_tex1 = textureLoad(buildings_tex, home_coords[1]);
            let h_tex2 = textureLoad(buildings_tex, home_coords[2]);
            let home_seg = h_tex1.w;
            let home_t = h_tex2.z;

            let target_b_coords = building_coords(work_id);
            let target_b_tex1 = textureLoad(buildings_tex, target_b_coords[1]);
            let target_b_tex2 = textureLoad(buildings_tex, target_b_coords[2]);
            let target_seg = u32(target_b_tex1.w);
            let target_t = target_b_tex2.z;

            // Start at the map edge
            let start_seg = params.entry_seg;
            let start_t = 0.0;

            let time_since_rent = rand(&rng_state) * 300.0;
            texel0 = vec4<f32>(50.0 + rand(&rng_state) * 450.0, time_since_rent, f32(work_id), f32(home_id));
            texel1 = vec4<f32>(f32(work_id), 0.0, -10.0, 0.0); // Travel, waiting for path
            texel2 = vec4<f32>(f32(start_seg), f32(start_seg), start_t, target_t);
            
            // Re-load variables for simulation
            money = texel0.x;

            // --- QUEUE INITIAL PATH REQUEST ---
            let req_idx = atomicAdd(&path_queue.count_x, 1u);
            let max_queue = 131072u;
            if req_idx < max_queue {
                path_queue.requests[req_idx] = PathRequest(u32(start_seg), target_seg, pid, 0u);
            }
            person_paths[pid * 512u] = 0xFFFFFFFFu;
            
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
        let logic_dt = params.dt * f32(params.cycle_frames);
        activity_time = activity_time - logic_dt;

        if activity_time <= 0.0 {
            // Activity done — pick next destination and apply economy.
            var current_building: u32 = 0u;
            if activity == ACT_HOME {
                current_building = u32(home);
                let rent = params.rent_cost;
                let tax = rent * params.tax_rent;
                money = max(0.0, money - (rent + tax));
                atomicAdd(&building_stats[current_building * 3u + 2u], u32(tax));
                atomicAdd(&stats.tax_rent_total, u32(tax));
            } else if activity == ACT_WORK {
                let salary = params.work_salary;
                let tax = salary * params.tax_income;
                money = money + (salary - tax);
                current_building = u32(work);
                atomicAdd(&building_stats[current_building * 3u + 2u], u32(tax));
                atomicAdd(&stats.tax_income_total, u32(tax));
            } else if activity == ACT_SHOP {
                let cost = params.shop_cost;
                let tax = cost * params.tax_consumption;
                money = max(0.0, money - (cost + tax));
                current_building = u32(destination);
                atomicAdd(&building_stats[current_building * 3u + 2u], u32(tax));
                atomicAdd(&stats.tax_consumption_total, u32(tax));
            }

            // Release building occupancy (Physically leaving)
            atomicSub(&building_stats[current_building * 3u], 1u);

            let current_b_coords = building_coords(current_building);
            let current_b_tex1 = textureLoad(buildings_tex, current_b_coords[1]);
            let current_b_tex2 = textureLoad(buildings_tex, current_b_coords[2]);
            let start_seg = u32(current_b_tex1.w); // road_seg is tex1.w
            let start_t = current_b_tex2.z;

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
            let next_b_tex2 = textureLoad(buildings_tex, next_b_coords[2]);
            let target_seg = u32(next_b_tex1.w);
            let target_t = next_b_tex2.z;

            // Check if start segment is full
            let start_r_coords = road_coords(start_seg);
            let start_r_tex1 = textureLoad(roads_tex, start_r_coords[1]);
            if start_r_tex1.x <= 0.15 {
                // Wait for space
                // Revert building occupancy (did not leave yet)
                atomicAdd(&building_stats[current_building * 3u], 1u);
                return;
            }

            // Queue path request
            let req_idx = atomicAdd(&path_queue.count_x, 1u);
            let max_queue = 131072u;
            if req_idx < max_queue {
                path_queue.requests[req_idx] = PathRequest(start_seg, target_seg, pid, 0u);
            }

            // Reset path buffer for this person
            let path_idx = pid * 512u;
            if path_idx < 67108864u { // 131072 * 512 = 67108864
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
            texel2.x = f32(start_seg);
            texel2.y = f32(start_seg);
            texel2.z = start_t;
            texel2.w = target_t;
            
            textureStore(people_tex, coords[0], texel0);
            textureStore(people_tex, coords[1], texel1);
            textureStore(people_tex, coords[2], texel2);
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

    // Read atomic values
    let occupants = f32(atomicLoad(&building_stats[bid * 3u]));
    let assigned = f32(atomicLoad(&building_stats[bid * 3u + 1u]));
    let tax_collected = f32(atomicLoad(&building_stats[bid * 3u + 2u]));

    let btype = tex0.z;

    var level = tex0.w;
    var capacity = tex1.z;

    var growth = tex2.x;
    var age_seconds = tex2.y;

    if capacity == 0.0 { return; }

    let logic_dt = params.dt * f32(params.cycle_frames);
    age_seconds = age_seconds + logic_dt;

    let fill = assigned / max(1.0, capacity);

    if fill > 0.7 && level < 4.0 {
        growth = growth + (0.1 * logic_dt);
        if growth >= 1.0 {
            level = level + 1.0;
            growth = 0.0;
            let base_cap = select(select(4.0, 6.0, btype == 1.0), 8.0, btype == 2.0);
            capacity = base_cap * exp2(level);
            // We no longer overwrite tex1.x with a fixed base_inc.
            // tex1.x = base_inc * (level + 1.0);
            tex1.z = capacity;
        }
    } else if fill < 0.05 && age_seconds > 60.0 {
        growth = growth - (0.02 * params.abandon_multiplier * logic_dt);
        if growth <= -1.0 {
            capacity = 0.0;
            tex1.z = 0.0;
        }
    } else {
        growth = min(0.0, max(-1.0, growth + (0.01 * logic_dt)));
    }

    tex0.w = level;
    tex1.x = tex1.x + tax_collected; // Accumulate collected tax
    tex1.y = occupants;
    tex2.x = growth;
    tex2.y = age_seconds;
    tex2.w = assigned;

    textureStore(buildings_tex, coords[0], tex0);
    textureStore(buildings_tex, coords[1], tex1);
    textureStore(buildings_tex, coords[2], tex2);
}

@compute @workgroup_size(64)
fn main_recount_stats(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slice_count = params.recount_slice_count;
    let slice_idx = params.recount_slice;
    
    let total_people = params.people_count;
    let slice_size = (total_people + slice_count - 1u) / slice_count;
    
    let slice_start_pid = slice_idx * slice_size;
    let slice_end_pid = min(total_people, (slice_idx + 1u) * slice_size);
    
    let chunk_size = 64u;
    let thread_start_pid = slice_start_pid + gid.x * chunk_size;
    
    if thread_start_pid >= slice_end_pid { return; }

    var p_count = 0u;
    var h_count = 0u;
    var w_count = 0u;
    var s_count = 0u;
    var t_count = 0u;
    var total_money = 0u;
    var bankrupt = 0u;

    for (var i = 0u; i < chunk_size; i = i + 1u) {
        let pid = thread_start_pid + i;
        if pid >= slice_end_pid { break; }

        let coords = person_coords(pid);
        let tex0 = textureLoad(people_tex, coords[0]);
        let tex1 = textureLoad(people_tex, coords[1]);

        let home_f = tex0.w;
        let work_f = tex1.x;
        let dest_f = tex0.z;

        if home_f < 0.0 && work_f < 0.0 { continue; } // Not spawned

        let home = u32(home_f);
        let work = u32(work_f);
        let dest = u32(dest_f);

        p_count = p_count + 1u;
        let activity = tex1.y;
        if activity == ACT_HOME { h_count = h_count + 1u; }
        else if activity == ACT_WORK { w_count = w_count + 1u; }
        else if activity == ACT_SHOP { s_count = s_count + 1u; }
        else if activity == ACT_TRAVEL { t_count = t_count + 1u; }
        else if activity == ACT_ARRIVED { t_count = t_count + 1u; }
        
        let money = tex0.x;
        total_money = total_money + u32(max(0.0, money));
        if money <= 0.0 { bankrupt = bankrupt + 1u; }

        if home_f >= 0.0 {
            atomicAdd(&building_stats[home * 3u + 1u], 1u);
        }
        if work_f >= 0.0 && work != home {
            atomicAdd(&building_stats[work * 3u + 1u], 1u);
        }

        if activity == ACT_HOME {
            if home_f >= 0.0 { atomicAdd(&building_stats[home * 3u], 1u); }
        } else if activity == ACT_WORK {
            if work_f >= 0.0 { atomicAdd(&building_stats[work * 3u], 1u); }
        } else if activity == ACT_SHOP {
            if dest_f >= 0.0 { atomicAdd(&building_stats[dest * 3u], 1u); }
        } else if activity == ACT_ARRIVED {
            if dest_f >= 0.0 { atomicAdd(&building_stats[dest * 3u], 1u); }
        }
    }

    atomicAdd(&stats.people_count, p_count);
    atomicAdd(&stats.home_count, h_count);
    atomicAdd(&stats.work_count, w_count);
    atomicAdd(&stats.shop_count, s_count);
    atomicAdd(&stats.travelling_count, t_count);
    atomicAdd(&stats.total_money, total_money);
    atomicAdd(&stats.bankrupt_count, bankrupt);
}

@compute @workgroup_size(64)
fn main_occupancy_gc_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.grid_w * params.grid_h { return; }
    atomicStore(&occupancy[idx], 0u);
}

@compute @workgroup_size(64)
fn main_occupancy_gc_repopulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.people_count { return; }

    let coords = person_coords(pid);
    var texel0 = textureLoad(people_tex, coords[0]);
    let texel1 = textureLoad(people_tex, coords[1]);
    let texel2 = textureLoad(people_tex, coords[2]);

    if texel0.x == 0.0 { return; } // Not spawned

    let activity = texel1.y;
    let current_seg = u32(texel2.x);
    let prev_seg = texel2.y;
    let activity_time = texel1.z;

    if activity == ACT_TRAVEL && current_seg != 0xFFFFFFFFu {
        let r_coords = road_coords(current_seg);
        let r_tex0 = textureLoad(roads_tex, r_coords[0]);
        let r_tex1 = textureLoad(roads_tex, r_coords[1]);
        let r_tex4 = textureLoad(roads_tex, r_coords[4]);
        let rtype = r_tex4.x;
        let seg_len = max(0.5, r_tex1.w);

        var start_at_b = false;
        if prev_seg >= 1000000.0 { start_at_b = true; }

        var frac = 0.0;
        if activity_time < 0.0 {
            frac = texel2.z; // start_t
        } else {
            let rem = clamp(activity_time / seg_len, 0.0, 1.0);
            if start_at_b { frac = rem; } else { frac = 1.0 - rem; }
        }

        let ax = r_tex0.x; let ay = r_tex0.y;
        let bx = r_tex0.z; let by = r_tex0.w;
        let dir = normalize(vec2<f32>(bx - ax, by - ay));
        let side = vec2<f32>(-dir.y, dir.x);
        
        var offset = select(0.35, -0.35, start_at_b);
        if rtype == 1.0 {
            let lane = f32(pid % 2u);
            let lane_offset = 0.5 + lane * 1.0;
            offset = select(lane_offset, -lane_offset, start_at_b);
        } else if rtype == 2.0 {
            let lane = f32(pid % 4u);
            let lane_offset = 0.5 + lane * 0.675;
            offset = select(lane_offset, -lane_offset, start_at_b);
        }
        
        let pos = mix(vec2<f32>(ax, ay), vec2<f32>(bx, by), frac) + side * offset;
        let tx = u32(pos.x + 0.5);
        let ty = u32(pos.y + 0.5);
        
        let idx = tx + ty * params.grid_w;
        if idx < params.grid_w * params.grid_h {
            atomicStore(&occupancy[idx], 1u);
            texel0.y = f32(idx);
            textureStore(people_tex, coords[0], texel0);
        }
    } else {
        // Not on road, ensure reserved_idx is reset
        if texel0.y != -1.0 {
            texel0.y = -1.0;
            textureStore(people_tex, coords[0], texel0);
        }
    }
}
