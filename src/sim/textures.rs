use bevy::{
    asset::RenderAssetUsages,
    image::Image,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};
use bytemuck::cast_slice;

use super::{buildings::BuildingData, people::PeopleData, roads::RoadData};

#[derive(Resource)]
pub struct DataTextures {
    pub people: Handle<Image>,
    pub roads: Handle<Image>,
    pub buildings: Handle<Image>,
    pub elevations: Handle<Image>,
}

pub fn create_data_textures(
    images: &mut Assets<Image>,
    people: &PeopleData,
    roads: &RoadData,
    buildings: &BuildingData,
) -> DataTextures {
    DataTextures {
        people: images.add(new_f32_texture(people.tex_width, people.tex_height, TextureFormat::Rgba32Float)),
        roads: images.add(new_f32_texture(roads.tex_width, roads.tex_height, TextureFormat::Rgba32Float)),
        buildings: images.add(new_f32_texture(buildings.tex_width, buildings.tex_height, TextureFormat::Rgba32Float)),
        elevations: images.add(new_f32_texture(128, 128, TextureFormat::R32Float)),
    }
}

fn new_f32_texture(w: u32, h: u32, format: TextureFormat) -> Image {
    let pixel_size = match format {
        TextureFormat::Rgba32Float => 16,
        TextureFormat::R32Float => 4,
        _ => 16,
    };
    let mut img = Image::new_fill(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &vec![0u8; pixel_size],
        format,
        RenderAssetUsages::default(),
    );
    // Storage binding is only needed once compute shaders are wired up; keep
    // the simpler usage for now so drivers without Rgba32Float storage support
    // (e.g. some Adreno variants) don't refuse the texture.
    img.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    img
}

/// Writes the dirty CPU shadows into the corresponding GPU textures.
/// Called each frame from the compute plugin.
pub fn upload_dirty_textures(
    images: &mut Assets<Image>,
    dt: &DataTextures,
    people: &mut PeopleData,
    roads: &mut RoadData,
    buildings: &mut BuildingData,
    grid: &crate::sim::grid::CityGrid,
) {
    if people.dirty {
        if let Some(img) = images.get_mut(&dt.people) {
            let bytes: &[u8] = cast_slice(&people.rows);
            fit_bytes(img, bytes);
        }
        people.dirty = false;
    }
    if roads.dirty {
        if let Some(img) = images.get_mut(&dt.roads) {
            let bytes: &[u8] = cast_slice(&roads.rows);
            fit_bytes(img, bytes);
        }
        roads.dirty = false;
    }
    if buildings.dirty {
        if let Some(img) = images.get_mut(&dt.buildings) {
            let bytes: &[u8] = cast_slice(&buildings.rows);
            fit_bytes(img, bytes);
        }
        buildings.dirty = false;
    }
    
    // Unconditionally upload grid elevations (it is a small 128x128 map)
    if let Some(img) = images.get_mut(&dt.elevations) {
        let bytes: &[u8] = cast_slice(&grid.elevations);
        fit_bytes(img, bytes);
    }
}

fn fit_bytes(img: &mut Image, src: &[u8]) {
    // src may be shorter than texture (capacity > live rows); pad with zero-init bytes.
    let Some(data) = img.data.as_mut() else { return; };
    let n = src.len().min(data.len());
    data[..n].copy_from_slice(&src[..n]);
    for b in &mut data[n..] {
        *b = 0;
    }
}
