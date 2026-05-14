use bevy::{
    asset::RenderAssetUsages,
    image::Image,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};

use super::{buildings::BuildingData, people::PeopleData, roads::RoadData};

#[derive(Resource)]
pub struct DataTextures {
    pub people: Handle<Image>,
    pub roads: Handle<Image>,
    pub buildings: Handle<Image>,
    pub elevations: Handle<Image>,
    pub road_points: Handle<Image>,
    pub car_transforms: Handle<Image>,
}

pub fn create_data_textures(
    images: &mut Assets<Image>,
    people: &PeopleData,
    roads: &RoadData,
    buildings: &BuildingData,
    grid_w: u32,
    grid_h: u32,
) -> DataTextures {
    // 4 texels per car (for a mat4x4). We can make it 1024 width.
    let transforms_width = 1024u32;
    let transforms_height = ((super::people::PEOPLE_CAPACITY * 4) + transforms_width - 1) / transforms_width;

    DataTextures {
        people: images.add(new_f32_texture(people.tex_width, people.tex_height, TextureFormat::Rgba32Float)),
        roads: images.add(new_f32_texture(roads.tex_width, roads.tex_height, TextureFormat::Rgba32Float)),
        buildings: images.add(new_f32_texture(buildings.tex_width, buildings.tex_height, TextureFormat::Rgba32Float)),
        elevations: images.add(new_f32_texture(grid_w, grid_h, TextureFormat::R32Float)),
        road_points: images.add(new_f32_texture(1024, 1024, TextureFormat::Rg32Float)),
        car_transforms: images.add(new_f32_texture(transforms_width, transforms_height, TextureFormat::Rgba32Float)),
    }
}

fn new_f32_texture(w: u32, h: u32, format: TextureFormat) -> Image {
    let pixel_size = match format {
        TextureFormat::Rgba32Float => 16,
        TextureFormat::R32Float => 4,
        TextureFormat::Rg32Float => 8,
        _ => 16,
    };
    let mut img = Image::new_fill(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &vec![0u8; pixel_size],
        format,
        RenderAssetUsages::default(),
    );
    img.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    img
}


