#import bevy_pbr::mesh_view_bindings::globals
#import bevy_pbr::forward_io::VertexOutput

// Fonction de pseudo-aléatoire (Hash 2D)
fn hash2d(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Bruit de valeur (Value Noise 2D)
fn noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    // Interpolation Hermite douce
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash2d(i);
    let b = hash2d(i + vec2<f32>(1.0, 0.0));
    let c = hash2d(i + vec2<f32>(0.0, 1.0));
    let d = hash2d(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Bruit fractal (Fractal Brownian Motion)
fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 4; i++) {
        value += amplitude * noise2d(p * freq);
        freq *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

@group(3) @binding(0) var grass_texture: texture_2d<f32>;
@group(3) @binding(1) var grass_sampler: sampler;
@group(3) @binding(2) var splat_texture: texture_2d<f32>;
@group(3) @binding(3) var splat_sampler: sampler;

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    // Les poids de splatting passés par le code Rust (Biomes)
    let weights = mesh.color;
    
    // On utilise la position dans le monde (X, Z)
    let pos = mesh.world_position.xz;

    // Échantillonnage de la Splat Map pour les routes
    let splat_dim = textureDimensions(splat_texture);
    let splat_coord = clamp(vec2<i32>(floor(pos)), vec2<i32>(0), vec2<i32>(splat_dim) - vec2<i32>(1));
    let splat = textureLoad(splat_texture, splat_coord, 0);
    let road_id = i32(splat.r * 255.0 + 0.5);
    let axis_val = splat.g * 255.0;
    
    // Échantillonnage de la texture d'herbe
    let grass_uv = pos * 1.5; 
    let grass_tex_color = textureSample(grass_texture, grass_sampler, grass_uv).rgb;

    // Génération de bruit pour les variations de couleur
    let n_base = fbm(pos * 0.1);
    let n_detail = fbm(pos * 1.0);

    let color_water = vec3<f32>(0.1, 0.3, 0.6) * (0.9 + 0.2 * n_detail);
    let color_plains = grass_tex_color * (0.85 + 0.3 * n_base);
    let color_forest = grass_tex_color * vec3<f32>(0.6, 0.8, 0.5) * (0.7 + 0.4 * fbm(pos * 0.5));
    let color_desert = vec3<f32>(0.8, 0.7, 0.5) * (0.9 + 0.2 * n_base);

    var final_color = color_water * weights.r +
                      color_plains * weights.g +
                      color_forest * weights.b +
                      color_desert * weights.a;

    // Rendu dynamique des routes
    if (road_id >= 1 && road_id <= 9) {
        let is_v = road_id == 2 || road_id == 5 || road_id == 8;
        let is_i = road_id == 3 || road_id == 6 || road_id == 9;
        
        // Coordonnées le long et à travers la route
        let pos_across = select(pos.y, pos.x, is_v);
        let pos_along = select(pos.x, pos.y, is_v);
        
        // Calcul de la distance depuis le centre exact de la route
        let pos_across_mod = pos_across - floor(pos_across / 256.0) * 256.0;
        let center = axis_val + 1.0;
        
        var signed_dist = pos_across_mod - center;
        if (signed_dist > 128.0) { signed_dist -= 256.0; }
        if (signed_dist < -128.0) { signed_dist += 256.0; }
        let dist_across = abs(signed_dist);
        
        let dash = fract(pos_along * 0.5) > 0.5; // Alternance des pointillés
        var road_color = vec3<f32>(0.15, 0.15, 0.17) + n_detail * 0.05;

        if (road_id >= 1 && road_id <= 3) { // Normal Road
            road_color = vec3<f32>(0.2, 0.2, 0.22) + n_detail * 0.05;
            if (!is_i) {
                if (dist_across < 0.05 && dash) {
                    road_color = vec3<f32>(0.8, 0.8, 0.8);
                }
            }
        } else if (road_id >= 4 && road_id <= 6) { // Highway (2x2)
            road_color = vec3<f32>(0.15, 0.15, 0.17) + n_detail * 0.05;
            if (!is_i) {
                if (dist_across > 1.8 && dist_across < 1.95) {
                    road_color = vec3<f32>(0.85, 0.85, 0.85); // Ligne de rive
                }
                if (dist_across > 0.05 && dist_across < 0.15) {
                    road_color = vec3<f32>(0.8, 0.6, 0.1); // Double jaune
                }
            }
        } else if (road_id >= 7 && road_id <= 9) { // Highway 2x4
            road_color = vec3<f32>(0.12, 0.12, 0.14) + n_detail * 0.05;
            if (!is_i) {
                if (dist_across > 2.8 && dist_across < 2.95) {
                    road_color = vec3<f32>(0.9, 0.9, 0.9); // Ligne de rive
                }
                if (dist_across < 0.15) {
                    road_color = vec3<f32>(0.7, 0.6, 0.1); // Terre-plein central
                }
                
                // Marquages des 4 voies par direction
                let is_lane_marker = abs(dist_across - 0.825) < 0.05 || 
                                     abs(dist_across - 1.5) < 0.05 || 
                                     abs(dist_across - 2.175) < 0.05;
                if (is_lane_marker && dash) {
                    road_color = vec3<f32>(0.8, 0.8, 0.8);
                }
            }
        }

        final_color = road_color;
    }

    // Output final
    return vec4<f32>(final_color, 1.0);
}
