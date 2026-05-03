//! GPU-driven car rendering.
//!
//! Spawns a single "Mega Mesh" containing copies of the car geometry for
//! every possible person in the simulation. The vertex shader (`car_instanced.wgsl`)
//! computes each car's world position directly from the GPU simulation textures
//! (`people_tex` and `roads_tex`), bypassing the Bevy CPU Transform layer entirely.

use bevy::{
    camera::visibility::NoFrustumCulling,
    prelude::*,
    render::render_resource::{AsBindGroup, PrimitiveTopology, ShaderType, VertexFormat},
    shader::ShaderRef,
};

use crate::sim::people::PEOPLE_CAPACITY;
use crate::sim::textures::DataTextures;

pub const ATTRIBUTE_CAR_ID: bevy::mesh::MeshVertexAttribute = bevy::mesh::MeshVertexAttribute::new("Vertex_CarId", 998, VertexFormat::Float32);

pub struct CarsRenderPlugin;

impl Plugin for CarsRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<CarInstancedMaterial>::default());
        app.init_resource::<CarsSetup>();
        app.add_systems(Update, setup_cars);
    }
}

#[derive(Resource, Default)]
struct CarsSetup(bool);

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CarInstancedMaterial {
    #[texture(0, dimension = "2d")]
    pub people_tex: Handle<Image>,
    #[texture(1, dimension = "2d")]
    pub roads_tex: Handle<Image>,
    #[texture(2, dimension = "2d")]
    pub elevations_tex: Handle<Image>,
    #[uniform(3)]
    pub params: CarMaterialParams,
}

#[derive(Clone, Default, ShaderType, Debug)]
pub struct CarMaterialParams {
    pub people_tex_w: u32,
    pub roads_tex_w: u32,
    pub grid_w: u32,
    pub pad: u32,
}

impl Material for CarInstancedMaterial {
    fn vertex_shader() -> ShaderRef { "shaders/car_instanced.wgsl".into() }
    fn fragment_shader() -> ShaderRef { "shaders/car_instanced.wgsl".into() }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
            ATTRIBUTE_CAR_ID.at_shader_location(3),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

fn setup_cars(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CarInstancedMaterial>>,
    dt: Option<Res<DataTextures>>,
    people: Option<Res<crate::sim::people::PeopleData>>,
    roads: Option<Res<crate::sim::roads::RoadData>>,
    grid: Option<Res<crate::sim::grid::CityGrid>>,
    mut setup: ResMut<CarsSetup>,
) {
    if setup.0 { return; }
    if let (Some(dt), Some(people), Some(roads), Some(grid)) = (dt, people, roads, grid) {
        let base_mesh = Cuboid::new(0.5, 0.3, 0.8).mesh().build();
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut car_ids: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let base_positions = base_mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().as_float3().unwrap();
        let base_normals = base_mesh.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap().as_float3().unwrap();
        
        // In Bevy 0.18, we can use try_into to convert to Vec<[f32; 2]>
        let base_uvs_vec: Vec<[f32; 2]> = base_mesh.attribute(Mesh::ATTRIBUTE_UV_0).unwrap().clone().try_into().expect("Expected Float32x2 UVs");
        
        let base_indices: Vec<u32> = match base_mesh.indices().unwrap() {
            bevy::mesh::Indices::U16(i) => i.iter().map(|&x| x as u32).collect(),
            bevy::mesh::Indices::U32(i) => i.clone(),
        };

        for car_id in 0..PEOPLE_CAPACITY {
            let vertex_offset = positions.len() as u32;
            positions.extend(base_positions);
            normals.extend(base_normals);
            uvs.extend(&base_uvs_vec);
            car_ids.extend(vec![car_id as f32; base_positions.len()]);

            for &idx in &base_indices {
                indices.push(idx + vertex_offset);
            }
        }

        let mut mega_mesh = Mesh::new(PrimitiveTopology::TriangleList, bevy::asset::RenderAssetUsages::default());
        mega_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mega_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mega_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mega_mesh.insert_attribute(ATTRIBUTE_CAR_ID, car_ids);
        mega_mesh.insert_indices(bevy::mesh::Indices::U32(indices));

        let material = materials.add(CarInstancedMaterial {
            people_tex: dt.people.clone(),
            roads_tex: dt.roads.clone(),
            elevations_tex: dt.elevations.clone(),
            params: CarMaterialParams {
                people_tex_w: people.tex_width,
                roads_tex_w: roads.tex_width,
                grid_w: grid.width,
                pad: 0,
            },
        });

        commands.spawn((
            Mesh3d(meshes.add(mega_mesh)),
            MeshMaterial3d(material),
            Transform::default(),
            NoFrustumCulling,
        ));

        setup.0 = true;
    }
}
