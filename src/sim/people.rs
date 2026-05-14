//! Per-person data laid out as an f32 texture.
//!
//! Each person = 2 RGBA32F texels (8 floats). Split into two texels because
//! the schema exceeds what fits in a single RGBA channel.
//!
//! Texel 0: [money, age, destination_building_id, home_building_id]
//! Texel 1: [work_building_id, activity_code, activity_time_remaining, path_cursor]
//!
//! activity_code: 0=travelling, 1=home, 2=work, 3=shopping.
//! activity_time_remaining: seconds left at current activity; 0 while travelling.
//! path_cursor: index into the person's cached path (see sim::roads).

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

pub const PEOPLE_CAPACITY: u32 = 524288;
pub const TEXELS_PER_PERSON: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PersonRow {
    pub money: f32,
    pub age: f32,
    pub destination: f32,
    pub home: f32,
    pub work: f32,
    pub activity_code: f32,
    pub activity_time: f32,
    pub path_cursor: f32,
    pub current_seg: f32,
    pub prev_seg: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Default for PersonRow {
    fn default() -> Self {
        Self {
            money: 0.0,
            age: 0.0,
            destination: -1.0,
            home: -1.0,
            work: -1.0,
            activity_code: 0.0,
            activity_time: 0.0,
            path_cursor: 0.0,
            current_seg: -1.0,
            prev_seg: -1.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

#[allow(dead_code)]
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Activity {
    Travelling = 0,
    Home = 1,
    Work = 2,
    Shopping = 3,
}

#[derive(Resource)]
pub struct PeopleData {
    /// Dense array; first `len` entries are live.
    pub rows: Vec<PersonRow>,
    pub len: u32,
    /// Texture size (texels). Packed as row-major: 4 texels per person across
    /// the width means width >= TEXELS_PER_PERSON * people_per_row. We use a
    /// simple square-ish layout.
    pub tex_width: u32,
    pub tex_height: u32,
    /// Dirty flag: true if CPU shadow needs to be re-uploaded.
    pub dirty: bool,
}

impl Default for PeopleData {
    fn default() -> Self {
        // Store PEOPLE_CAPACITY persons; each uses TEXELS_PER_PERSON texels.
        // Layout: width covers TEXELS_PER_PERSON * persons_per_row; keep square.
        let total_texels = PEOPLE_CAPACITY * TEXELS_PER_PERSON;
        let side = (total_texels as f32).sqrt().ceil() as u32;
        // Round width to multiple of TEXELS_PER_PERSON so rows don't split a person.
        let width = ((side + TEXELS_PER_PERSON - 1) / TEXELS_PER_PERSON) * TEXELS_PER_PERSON;
        let height = (total_texels + width - 1) / width;
        Self {
            rows: vec![PersonRow::default(); PEOPLE_CAPACITY as usize],
            len: 0,
            tex_width: width,
            tex_height: height,
            dirty: true,
        }
    }
}

