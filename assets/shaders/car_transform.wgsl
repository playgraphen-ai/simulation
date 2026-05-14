#import bevy_pbr::mesh_view_bindings::view

struct CarMaterialParams {
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
    do_occupancy_gc: u32,
}

@group(0) @binding(0) var people_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var roads_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var<uniform> params: CarMaterialParams;
@group(0) @binding(10) var road_points_tex: texture_2d<f32>;
@group(0) @binding(11) var elevations_tex: texture_2d<f32>;
@group(0) @binding(12) var car_transforms_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(13) var elevations_sampler: sampler;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pid = global_id.x;
    if pid >= params.people_count { return; }

    let base = i32(pid * 3u);
    let pw = i32(params.people_tex_w);
    let c0 = vec2<i32>(base % pw, base / pw);
    let c1 = vec2<i32>((base + 1) % pw, (base + 1) / pw);
    let c2 = vec2<i32>((base + 2) % pw, (base + 2) / pw);

    let tex1 = textureLoad(people_tex, c1);
    let tex2 = textureLoad(people_tex, c2);

    let activity = tex1.y;
    let activity_time = tex1.z;
    let current_seg = u32(tex2.x);
    let prev_seg_raw = tex2.y;

    var world_pos = vec3<f32>(0.0, -10000.0, 0.0);
    var rot_matrix = mat3x3<f32>(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    );
    // Flag for coloring: 0.0 = normal, 1.0 = white (abandoning)
    var is_white = 0.0;

    if (activity == 0.0 || activity == 4.0) && current_seg != 0xFFFFFFFFu {
        if activity == 4.0 { is_white = 1.0; }
        let rw = i32(params.roads_tex_w);
        
        let rbase = i32(current_seg * 5u);
        let rc0 = vec2<i32>(rbase % rw, rbase / rw);
        let rc1 = vec2<i32>((rbase + 1) % rw, (rbase + 1) / rw);
        let rc4 = vec2<i32>((rbase + 4) % rw, (rbase + 4) / rw);
        let rtex0 = textureLoad(roads_tex, rc0);
        let rtex1 = textureLoad(roads_tex, rc1);
        let rtex4 = textureLoad(roads_tex, rc4);
        let rtype = rtex4.x;
        
        let seg_len = max(0.01, rtex1.w);
        let links_offset = u32(rtex1.y);
        let links_count = u32(rtex1.z);
        
        var start_at_b = false;
        if prev_seg_raw >= 1000000.0 {
            start_at_b = true;
        }

        var frac = 0.0;
        if activity_time < 0.0 {
            frac = tex2.z;
        } else {
            var time_val = max(0.0, activity_time);
            var remaining_frac = clamp(time_val / seg_len, 0.0, 1.0);
            
            if start_at_b {
                frac = remaining_frac;
            } else {
                frac = 1.0 - remaining_frac;
            }
        }

        let safe_links_count = max(1u, links_count);
        let target_link_f = frac * f32(safe_links_count - 1u);
        let i = u32(floor(target_link_f));
        let f = fract(target_link_f);
        
        let rw_pts = 1024u;
        let idx0 = links_offset + i;
        let p_c0 = vec2<i32>(i32(idx0 % rw_pts), i32(idx0 / rw_pts));
        let idx1 = links_offset + min(i + 1u, links_count - 1u);
        let p_c1 = vec2<i32>(i32(idx1 % rw_pts), i32(idx1 / rw_pts));
        
        let p0 = textureLoad(road_points_tex, p_c0, 0).xy;
        let p1 = textureLoad(road_points_tex, p_c1, 0).xy;

        let x = mix(p0.x, p1.x, f) + 0.5;
        let z = mix(p0.y, p1.y, f) + 0.5;

        let el_a = textureSampleLevel(elevations_tex, elevations_sampler, vec2<f32>(p0.x + 0.5, p0.y + 0.5) / f32(params.grid_w), 0.0).x;
        let el_b = textureSampleLevel(elevations_tex, elevations_sampler, vec2<f32>(p1.x + 0.5, p1.y + 0.5) / f32(params.grid_w), 0.0).x;
        
        let uv = vec2<f32>(x + 0.5, z + 0.5) / f32(params.grid_w);
        let terrain_h = textureSampleLevel(elevations_tex, elevations_sampler, uv, 0.0).x;
        let y = terrain_h + 0.15;

        world_pos = vec3<f32>(x, y, z);

        var dir = vec3<f32>(p1.x - p0.x, el_b - el_a, p1.y - p0.y);
        if start_at_b {
            dir = -dir;
        }
        
        let len_sq = dot(dir, dir);
        if len_sq > 0.001 {
            let d = normalize(dir);
            let up = vec3<f32>(0.0, 1.0, 0.0);
            let right = normalize(cross(up, d));
            
            var offset = 0.35;
            if rtype == 1.0 {
                let lane = f32(pid % 2u);
                offset = 0.5 + lane * 1.0;
            } else if rtype == 2.0 {
                let lane = f32(pid % 4u);
                offset = 0.5 + lane * 0.675;
            }
            world_pos = world_pos + right * offset;

            let real_up = cross(d, right);
            rot_matrix = mat3x3<f32>(
                right.x, real_up.x, d.x,
                right.y, real_up.y, d.y,
                right.z, real_up.z, d.z
            );
        }
    }

    let t_w = 1024u;
    let t_idx = pid * 2u;
    
    let tx0 = t_idx % t_w;
    let ty0 = t_idx / t_w;
    textureStore(car_transforms_tex, vec2<i32>(i32(tx0), i32(ty0)), vec4<f32>(world_pos.x, world_pos.y, world_pos.z, is_white));
    
    let tx1 = (t_idx + 1u) % t_w;
    let ty1 = (t_idx + 1u) / t_w;
    textureStore(car_transforms_tex, vec2<i32>(i32(tx1), i32(ty1)), vec4<f32>(rot_matrix[2].x, rot_matrix[2].z, rot_matrix[0].x, rot_matrix[0].z));
}
