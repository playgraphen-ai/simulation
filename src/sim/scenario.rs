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
    let block_size = 8;
    let grid_size_x = settings.grid_x;
    let grid_size_y = settings.grid_y;
    
    // Define 4 distant origin points for the 4 cities, now 3x closer (200 apart instead of 600)
    let offsets: [(u32, u32); 4] = [
        (200, 200),
        (200, 400),
        (400, 200),
        (400, 400),
    ];

    let mut first_building_id = None;
    let mid_gy = grid_size_y / 2;

    for &(start_x, start_y) in &offsets {
        // 1. Build the grid of roads
        // Horizontal roads
        for gy in 0..=grid_size_y {
            let y = start_y + gy * block_size;
            let mut prev = None;
            
            // For the middle horizontal road, extend it slightly to the left as an entry point
            let row_start_x = if gy == mid_gy { start_x.saturating_sub(10) } else { start_x };
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
    }

    // Connect the 4 cities with single roads
    let c0_mid_y = offsets[0].1 + mid_gy * block_size;
    let c1_mid_y = offsets[1].1 + mid_gy * block_size;
    let c0_end_x = offsets[0].0 + grid_size_x * block_size;
    let c2_start_x = offsets[2].0;
    
    // Connect top-left (0) to top-right (2)
    for x in c0_end_x..=c2_start_x {
        roads.push_link((x, c0_mid_y), (x.saturating_sub(1).max(c0_end_x), c0_mid_y));
    }
    // Connect bottom-left (1) to bottom-right (3)
    let c1_end_x = offsets[1].0 + grid_size_x * block_size;
    let c3_start_x = offsets[3].0;
    for x in c1_end_x..=c3_start_x {
        roads.push_link((x, c1_mid_y), (x.saturating_sub(1).max(c1_end_x), c1_mid_y));
    }
    
    // Connect top-left (0) to bottom-left (1)
    let c0_mid_x = offsets[0].0 + (grid_size_x / 2) * block_size;
    let c0_end_y = offsets[0].1 + grid_size_y * block_size;
    let c1_start_y = offsets[1].1;
    for y in c0_end_y..=c1_start_y {
        roads.push_link((c0_mid_x, y), (c0_mid_x, y.saturating_sub(1).max(c0_end_y)));
    }
    
    // Connect top-right (2) to bottom-right (3)
    let c2_mid_x = offsets[2].0 + (grid_size_x / 2) * block_size;
    let c2_end_y = offsets[2].1 + grid_size_y * block_size;
    let c3_start_y = offsets[3].1;
    for y in c2_end_y..=c3_start_y {
        roads.push_link((c2_mid_x, y), (c2_mid_x, y.saturating_sub(1).max(c2_end_y)));
    }

    let tile_to_seg = roads.rebuild_topology();
    let mut entry_seg = 0;
    
    let entry_x = offsets[0].0.saturating_sub(10);
    let entry_y = offsets[0].1 + mid_gy * block_size;

    for ((tx, ty), seg_id) in tile_to_seg {
        grid.set(tx, ty, Tile::Road(seg_id));
        if tx == entry_x && ty == entry_y {
            entry_seg = seg_id;
        }
    }

    commands.insert_resource(ScenarioLayout { entry_seg });

    // 2. Zone the blocks for all 4 cities
    for &(start_x, start_y) in &offsets {
        for gy in 0..grid_size_y {
            for gx in 0..grid_size_x {
                // Pseudo-random distribution based on coordinates to get 40% res, 30% office, 30% shop
                let pseudo_rand = (gx * 7 + gy * 13 + start_x + start_y) % 10;
                let zone = match pseudo_rand {
                    0..=3 => ZoneType::Residential, // 40%
                    4..=6 => ZoneType::Office,      // 30%
                    _ => ZoneType::Shop,            // 30%
                };

                let bx = start_x + gx * block_size;
                let by = start_y + gy * block_size;

                for dy in 1..block_size {
                    for dx in 1..block_size {
                        let tx = bx + dx;
                        let ty = by + dy;
                        
                        if dx == 1 || dx == block_size - 1 || dy == 1 || dy == block_size - 1 {
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
    }

    if let Some(bid) = first_building_id {
        commands.insert_resource(Selection {
            obj: Some(SelectedObj::Building(bid)),
        });
    }

    grid.set_changed();

    spawn_ev.write(SpawnPeopleRequest { count: settings.population });

    info!(
        "4 cities built, each {}x{} blocks, connected. {} people spawned.",
        grid_size_x, grid_size_y, settings.population
    );
}
