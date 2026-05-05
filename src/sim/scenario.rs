//! Starter scenario: builds a test city at startup so we can iterate without
//! click-painting every run. Layout:
//!
//!   - one long horizontal trunk road across the middle of the map
//!   - three perpendicular side streets ("Residential", "Office", "Shop")
//!     branching north off the trunk
//!   - every tile adjacent to a side street is zoned with that street's type,
//!     and the road tool auto-materialises buildings on both sides
//!   - N people spawned straight away
//!
//! Runs once at Startup after the grid is reset.

use bevy::prelude::*;

use super::{
    buildings::{spawn_building, BuildingData},
    grid::{CityGrid, Tile, ZoneType},
    roads::RoadData,
    SpawnPeopleRequest,
};
use crate::ui::inspector::{Selection, SelectedObj};

/// Tile coordinates of the trunk and side-streets so other systems can
/// reference them if needed.
#[derive(Resource, Debug, Clone, Copy)]
pub struct ScenarioLayout {
    pub entry_seg: u32,
}

pub fn build_starter_scenario(
    mut grid: ResMut<CityGrid>,
    mut roads: ResMut<RoadData>,
    mut buildings: ResMut<BuildingData>,
    mut commands: Commands,
    mut spawn_ev: MessageWriter<SpawnPeopleRequest>,
    settings: Res<crate::ui::menu::MenuSettings>,
) {
    let w = grid.width;
    let h = grid.height;
    
    // We'll build a grid of blocks.
    // Each block is roughly 10x10 tiles.
    let block_size = 8;
    let grid_size_x = settings.grid_x;
    let grid_size_y = settings.grid_y;
    
    let start_x = (w - (grid_size_x * block_size)) / 2;
    let start_y = (h - (grid_size_y * block_size)) / 2;

    let mut first_building_id = None;

    // 1. Build the grid of roads
    // Horizontal roads
    let mid_gy = grid_size_y / 2;
    for gy in 0..=grid_size_y {
        let y = start_y + gy * block_size;
        let mut prev = None;
        
        let row_start_x = if gy == mid_gy { 0 } else { start_x };
        let row_end_x = start_x + grid_size_x * block_size;

        for x in row_start_x..=row_end_x {
            if grid.get(x, y) == Some(Tile::Water) {
                prev = None; // break the road
                continue;
            }
            let here = (x, y);
            if let Some(other) = prev {
                roads.push_link(here, other);
            } else {
                roads.push_link(here, here);
            }
            prev = Some(here);
        }
    }
    // Vertical roads
    for gx in 0..=grid_size_x {
        let x = start_x + gx * block_size;
        let mut prev = None;
        for y in start_y..=(start_y + grid_size_y * block_size) {
            if grid.get(x, y) == Some(Tile::Water) {
                prev = None;
                continue;
            }
            let here = (x, y);
            if let Some(other) = prev {
                roads.push_link(here, other);
            } else {
                roads.push_link(here, here);
            }
            prev = Some(here);
        }
    }

    let tile_to_seg = roads.rebuild_topology();
    let mut entry_seg = 0;
    for ((tx, ty), seg_id) in tile_to_seg {
        grid.set(tx, ty, Tile::Road(seg_id));
        if tx == 0 && ty == start_y + mid_gy * block_size {
            entry_seg = seg_id;
        }
    }

    commands.insert_resource(ScenarioLayout { entry_seg });

    // 2. Zone the blocks
    for gy in 0..grid_size_y {
        for gx in 0..grid_size_x {
            // Determine zone type for this block based on position
            // Center is mostly Offices/Shops, periphery is Residential
            let dist_from_center = ((gx as i32 - 3).abs() + (gy as i32 - 2).abs()) as f32;
            let zone = if dist_from_center < 1.5 {
                ZoneType::Shop
            } else if dist_from_center < 3.5 {
                ZoneType::Office
            } else {
                ZoneType::Residential
            };

            // Fill the interior of the block (not the roads)
            let bx = start_x + gx * block_size;
            let by = start_y + gy * block_size;

            for dy in 1..block_size {
                for dx in 1..block_size {
                    let tx = bx + dx;
                    let ty = by + dy;
                    
                    // We only want to zone near the roads for realism/utility
                    if dx == 1 || dx == block_size - 1 || dy == 1 || dy == block_size - 1 {
                        // Find the nearest road segment for this building
                        let mut nearest_road = None;
                        for (nx, ny) in grid.neighbours4(tx, ty) {
                            if let Some(Tile::Road(sid)) = grid.get(nx, ny) {
                                nearest_road = Some((sid, (nx, ny)));
                                break;
                            }
                        }

                        if let Some((seg_id, road_tile)) = nearest_road {
                            grid.set(tx, ty, Tile::Zone(zone));
                            let road_t = roads.get_tile_t(seg_id, road_tile);
                            if let Some(bid) = spawn_building(&mut buildings, &mut grid, (tx, ty), zone, seg_id, road_t) {
                                if first_building_id.is_none() {
                                    first_building_id = Some(bid);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(bid) = first_building_id {
        commands.insert_resource(Selection {
            obj: Some(SelectedObj::Building(bid)),
        });
    }

    grid.set_changed();

    // Seed the city with a lot of people to make it feel alive!
    spawn_ev.write(SpawnPeopleRequest { count: settings.population });

    info!(
        "Large starter city built: {}x{} blocks, {} people spawned.",
        grid_size_x, grid_size_y, settings.population
    );
}
