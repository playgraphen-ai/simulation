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
    // On utilise floor(pos) pour que la route soit parfaitement alignée sur la grille
    let splat_dim = textureDimensions(splat_texture);
    let splat_coord = clamp(vec2<i32>(floor(pos)), vec2<i32>(0), vec2<i32>(splat_dim) - vec2<i32>(1));
    // On utilise textureLoad pour une lecture brute sans interpolation (pixel perfect)
    let road_id_f = textureLoad(splat_texture, splat_coord, 0).r;
    let road_id = i32(road_id_f * 255.0 + 0.5);
    
    // Échantillonnage de la texture d'herbe
    // On augmente la fréquence pour que l'herbe soit détaillée à l'échelle d'une voiture
    // Si une case (1.0 unité) est une voiture (~4m), pos * 2.0 répète la texture 2 fois par case.
    let grass_uv = pos * 1.5; 
    let grass_tex_color = textureSample(grass_texture, grass_sampler, grass_uv).rgb;

    // Génération de bruit pour les variations de couleur
    let n_base = fbm(pos * 0.1);   // Variations larges (biomes/terrain)
    let n_detail = fbm(pos * 1.0); // Variations fines (détails au sol)

    // Définition des couleurs de base, modulées par le bruit et la texture
    let color_water = vec3<f32>(0.1, 0.3, 0.6) * (0.9 + 0.2 * n_detail);
    let color_plains = grass_tex_color * (0.85 + 0.3 * n_base);
    let color_forest = grass_tex_color * vec3<f32>(0.6, 0.8, 0.5) * (0.7 + 0.4 * fbm(pos * 0.5));
    let color_desert = vec3<f32>(0.8, 0.7, 0.5) * (0.9 + 0.2 * n_base);

    // Mélange final via les poids de splatting
    var final_color = color_water * weights.r +
                      color_plains * weights.g +
                      color_forest * weights.b +
                      color_desert * weights.a;

    // Couleur de la route (Gris foncé bitume)
    // On ajoute un peu de bruit de détail sur la route pour qu'elle ne soit pas plate
    let road_noise = n_detail * 0.05;
    var road_color = vec3<f32>(0.15, 0.15, 0.17) + road_noise;
    
    // Marquages au sol pour les autoroutes
    if (road_id == 2) { // Highway
        let local_pos = fract(pos);
        // Ligne blanche discontinue sur les bords ou milieu ?
        // Ici on va faire une ligne blanche sur les bords des cases 2x2
        let is_edge = local_pos.x < 0.05 || local_pos.x > 0.95 || local_pos.y < 0.05 || local_pos.y > 0.95;
        if (is_edge) {
            road_color += vec3<f32>(0.1, 0.1, 0.1);
        }
    } else if (road_id == 3) { // Highway 2x4
        let local_pos = fract(pos);
        road_color = vec3<f32>(0.12, 0.12, 0.14) + road_noise; // Un peu plus sombre
        
        // Lignes jaunes doubles au milieu d'une des cases
        let dist_x = abs(local_pos.x - 0.5);
        let dist_y = abs(local_pos.y - 0.5);
        if ((dist_x < 0.03 && dist_x > 0.01) || (dist_y < 0.03 && dist_y > 0.01)) {
            road_color = vec3<f32>(0.8, 0.6, 0.1);
        }
        
        // Lignes blanches pointillées pour les voies
        let lane_x = fract(pos.x * 2.0);
        let lane_y = fract(pos.y * 2.0);
        let dash = fract(pos.x * 0.5) > 0.5 || fract(pos.y * 0.5) > 0.5;
        if ((abs(lane_x - 0.5) < 0.01 || abs(lane_y - 0.5) < 0.01) && dash) {
            road_color = vec3<f32>(0.7, 0.7, 0.7);
        }
    }

    // On mix la route au dessus du terrain
    if (road_id > 0) {
        final_color = road_color;
    }

    // Output final
    return vec4<f32>(final_color, 1.0);
}
