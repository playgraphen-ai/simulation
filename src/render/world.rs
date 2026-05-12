//! Draws the ground, zoned tiles, roads and buildings.
//!
//! Buildings use GLTF scenes from the Kenney City Kit (CC0). Roads and zone
//! decals stay as simple plane meshes for now.

use bevy::prelude::*;

use crate::sim::{
    buildings::{BuildingData, MAX_LEVEL},
    grid::{CityGrid, Tile, ZoneType},
    roads::RoadData,
};

pub struct WorldRenderPlugin;

impl Plugin for WorldRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldVisuals>()
            .add_systems(Startup, (setup_ground.after(crate::sim::startup), build_palette))
            .add_systems(Update, (sync_zone_view, sync_road_view, sync_building_view, sync_splat_map_system));
    }
}

#[derive(Resource, Default)]
pub struct WorldVisuals {
    pub tile_mesh: Handle<Mesh>,
    pub road_mesh: Handle<Mesh>,
    pub mat_zone_res: Handle<StandardMaterial>,
    pub mat_zone_off: Handle<StandardMaterial>,
    pub mat_zone_shop: Handle<StandardMaterial>,
    pub mat_road: Handle<StandardMaterial>,
    /// Per (btype, level) GLTF scene handle.
    pub residential_scenes: Vec<Handle<Scene>>, // 5
    pub office_scenes: Vec<Handle<Scene>>,      // 5
    pub shop_scenes: Vec<Handle<Scene>>,        // 5
    pub road_scene: Handle<Scene>,
    pub car_scene: Handle<Scene>,
}

#[derive(Component)]
struct ZoneMarker;

#[derive(Component)]
struct RoadMarker;

#[derive(Component)]
pub struct BuildingMarker(pub u32, pub u32); // (id, level)

use bevy::render::render_resource::PrimitiveTopology;
use bevy::mesh::Indices;
use bevy::asset::RenderAssetUsages;
use crate::sim::grid::Biome;
use super::terrain::TerrainMaterial;

use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};

#[derive(Resource)]
pub struct TerrainSplatMap(pub Handle<Image>);

fn setup_ground(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    grid: Res<CityGrid>,
) {
    let w = grid.width;
    let h = grid.height;
    
    // Create the splat map image
    let mut splat_image = Image::new_fill(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    splat_image.texture_descriptor.usage |= TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    let splat_handle = images.add(splat_image);
    commands.insert_resource(TerrainSplatMap(splat_handle.clone()));

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((w * h) as usize);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((w * h) as usize);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity((w * h) as usize);
    let mut indices: Vec<u32> = Vec::with_capacity(((w - 1) * (h - 1) * 6) as usize);
    
    // Generate vertices
    for y in 0..h {
        for x in 0..w {
            let i = grid.idx(x, y);
            let elev = grid.elevations[i];
            // Remove the 0.5 offset to align vertices with grid corners
            positions.push([x as f32, elev, y as f32]);
            
            // Encode biomes as splat weights: Water(R), Plains(G), Forest(B), Desert(A)
            let color = match grid.biomes[i] {
                Biome::Water => [1.0, 0.0, 0.0, 0.0],
                Biome::Plains => [0.0, 1.0, 0.0, 0.0],
                Biome::Forest => [0.0, 0.0, 1.0, 0.0],
                Biome::Desert => [0.0, 0.0, 0.0, 1.0],
            };
            colors.push(color);
            normals.push([0.0, 1.0, 0.0]); // Simple upward normals for flat shading look
        }
    }
    
    // Compute basic normals
    for y in 0..(h - 1) {
        for x in 0..(w - 1) {
            let top_left = y * w + x;
            let top_right = top_left + 1;
            let bottom_left = (y + 1) * w + x;
            let bottom_right = bottom_left + 1;
            
            // Triangle 1
            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);
            
            // Triangle 2
            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }
    
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_normals(); // Let Bevy compute smooth normals
    
    let mat = materials.add(TerrainMaterial {
        grass: asset_server.load("textures/grass.png"),
        splat_map: splat_handle,
    });
    
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(mat),
        Transform::default(),
    ));
}

fn build_palette(
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut vis: ResMut<WorldVisuals>,
) {
    vis.tile_mesh = meshes.add(Plane3d::default().mesh().size(0.96, 0.96));
    vis.road_mesh = meshes.add(Plane3d::default().mesh().size(0.96, 0.96));
    vis.mat_zone_res = materials.add(tint(0.2, 0.7, 0.25, 0.4));
    vis.mat_zone_off = materials.add(tint(0.25, 0.45, 0.85, 0.4));
    vis.mat_zone_shop = materials.add(tint(0.9, 0.65, 0.2, 0.4));
    vis.mat_road = materials.add(StandardMaterial {
        base_color: Color::srgb(0.15, 0.15, 0.18),
        perceptual_roughness: 0.9,
        ..default()
    });
    vis.residential_scenes = (0..=MAX_LEVEL)
        .map(|lvl| asset_server.load(GltfAssetLabel::Scene(0)
            .from_asset(format!("models/buildings/residential_{}.glb", lvl))))
        .collect();
    vis.office_scenes = (0..=MAX_LEVEL)
        .map(|lvl| asset_server.load(GltfAssetLabel::Scene(0)
            .from_asset(format!("models/buildings/office_{}.glb", lvl))))
        .collect();
    vis.shop_scenes = (0..=MAX_LEVEL)
        .map(|lvl| asset_server.load(GltfAssetLabel::Scene(0)
            .from_asset(format!("models/buildings/shop_{}.glb", lvl))))
        .collect();
    vis.road_scene = asset_server.load(
        GltfAssetLabel::Scene(0).from_asset("models/roads/road_straight.glb"));
    vis.car_scene = asset_server.load(
        GltfAssetLabel::Scene(0).from_asset("models/vehicles/sedan.glb"));
}

fn tint(r: f32, g: f32, b: f32, a: f32) -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgba(r, g, b, a),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    }
}

fn sync_zone_view(
    mut commands: Commands,
    grid: Res<CityGrid>,
    vis: Res<WorldVisuals>,
    existing: Query<Entity, With<ZoneMarker>>,
) {
    if !grid.is_changed() { return; }
    for e in &existing {
        commands.entity(e).despawn();
    }
    for y in 0..grid.height {
        for x in 0..grid.width {
            if let Some(Tile::Zone(z)) = grid.get(x, y) {
                let mat = match z {
                    ZoneType::Residential => vis.mat_zone_res.clone(),
                    ZoneType::Office => vis.mat_zone_off.clone(),
                    ZoneType::Shop => vis.mat_zone_shop.clone(),
                };
                let elev = grid.elevations[grid.idx(x, y)];
                commands.spawn((
                    Mesh3d(vis.tile_mesh.clone()),
                    MeshMaterial3d(mat),
                    Transform::from_xyz(x as f32 + 0.5, elev + 0.02, y as f32 + 0.5),
                    ZoneMarker,
                ));
            }
        }
    }
}

fn sync_road_view(
    mut commands: Commands,
    roads: Res<RoadData>,
    existing: Query<Entity, With<RoadMarker>>,
) {
    if !roads.is_changed() { return; }
    // We no longer spawn GLTF meshes for roads, because we render them directly
    // on the terrain using the splat map. So we just ensure any legacy road 
    // marker entities are despawned.
    for e in &existing {
        commands.entity(e).despawn();
    }
}

fn sync_building_view(
    mut commands: Commands,
    buildings: Res<BuildingData>,
    grid: Res<CityGrid>,
    vis: Res<WorldVisuals>,
    mut existing: Query<(Entity, &mut BuildingMarker, &mut SceneRoot, &mut Transform)>,
) {
    if !buildings.is_changed() { return; }
    let mut seen = std::collections::HashSet::new();
    for (e, mut marker, mut scene, mut _tf) in &mut existing {
        let id = marker.0 as usize;
        if id >= buildings.items.len() {
            commands.entity(e).despawn();
            continue;
        }
        let b = &buildings.items[id];
        if b.capacity == 0 {
            commands.entity(e).despawn();
            continue;
        }
        if marker.1 != b.level {
            scene.0 = scene_for(&vis, b.btype, b.level);
            marker.1 = b.level;
        }
        seen.insert(marker.0);
    }
    for (id, b) in buildings.items.iter().enumerate() {
        let id_u32 = id as u32;
        if seen.contains(&id_u32) || b.capacity == 0 { continue; }
        let elev = grid.elevations[grid.idx(b.tile.0, b.tile.1)];
        let size = match b.btype {
            ZoneType::Residential => 3.0,
            _ => 4.0,
        };
        let offset = size / 2.0;

        commands.spawn((
            SceneRoot(scene_for(&vis, b.btype, b.level)),
            Transform::from_xyz(b.tile.0 as f32 + offset, elev, b.tile.1 as f32 + offset)
                .with_scale(Vec3::splat(size)),
            BuildingMarker(id_u32, b.level),
        ));
    }
}

fn scene_for(vis: &WorldVisuals, bt: ZoneType, lvl: u32) -> Handle<Scene> {
    let lvl = (lvl as usize).min(MAX_LEVEL as usize);
    match bt {
        ZoneType::Residential => vis.residential_scenes[lvl].clone(),
        ZoneType::Office => vis.office_scenes[lvl].clone(),
        ZoneType::Shop => vis.shop_scenes[lvl].clone(),
    }
}

fn sync_splat_map_system(
    grid: Res<CityGrid>,
    roads: Res<RoadData>,
    splat_map_res: Option<Res<TerrainSplatMap>>,
    mut images: ResMut<Assets<Image>>,
) {
    if !grid.is_changed() { return; }
    if let Some(res) = splat_map_res {
        if let Some(img) = images.get_mut(&res.0) {
            let mut data = vec![0u8; (grid.width * grid.height * 4) as usize];
            for y in 0..grid.height {
                for x in 0..grid.width {
                    if let Some(Tile::Road(seg_id)) = grid.get(x, y) {
                        let seg = &roads.segments[seg_id as usize];
                        let rtype = seg.road_type;
                        
                        // Determine actual direction from segment endpoints
                        let dx = (seg.b.0 as i32 - seg.a.0 as i32).abs();
                        let dy = (seg.b.1 as i32 - seg.a.1 as i32).abs();
                        let is_v = dy > dx;
                        
                        // Detect if tile is near the segment ends (junctions)
                        let radius = match rtype {
                            crate::sim::roads::RoadType::Highway2x4 => 3,
                            crate::sim::roads::RoadType::Highway => 2,
                            crate::sim::roads::RoadType::Normal => 1,
                        };
                        
                        let dist_a = (x as i32 - seg.a.0 as i32).abs().max((y as i32 - seg.a.1 as i32).abs());
                        let dist_b = (x as i32 - seg.b.0 as i32).abs().max((y as i32 - seg.b.1 as i32).abs());
                        let is_i = dist_a <= radius || dist_b <= radius;

                        let variant = if is_i {
                            2 // Intersection
                        } else if is_v {
                            1 // Vertical
                        } else {
                            0 // Horizontal
                        };

                        let base = match rtype {
                            crate::sim::roads::RoadType::Normal => 1,
                            crate::sim::roads::RoadType::Highway => 4,
                            crate::sim::roads::RoadType::Highway2x4 => 7,
                        };
                        
                        let idx = (y * grid.width + x) as usize * 4;
                        data[idx] = base + variant; // R: Type and Direction
                        data[idx + 1] = if is_v { (seg.a.0 % 256) as u8 } else { (seg.a.1 % 256) as u8 }; // G: Center axis anchor
                        data[idx + 2] = 0; // B
                        data[idx + 3] = 255; // A
                    }
                }
            }
            img.data = Some(data);
        }
    }
}
