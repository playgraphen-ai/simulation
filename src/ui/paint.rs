//! "Painting" system: when a tool is active and the user left-clicks (and
//! holds) over a grid tile that is not covered by UI, stamp the tool output
//! there. For road tool, also auto-materialize buildings on adjacent zoned
//! tiles.

use bevy::prelude::*;

use crate::render::camera::CursorTile;
use crate::sim::buildings::{spawn_building, BuildingData};
use crate::sim::grid::{CityGrid, Tile, ZoneType};
use crate::sim::roads::{RoadData, RoadType};

use super::tools::ActiveTool;

pub fn paint_tick_system(
    mouse: Res<ButtonInput<MouseButton>>,
    cursor: Res<CursorTile>,
    active: Res<ActiveTool>,
    mut grid: ResMut<CityGrid>,
    mut roads: ResMut<RoadData>,
    mut buildings: ResMut<BuildingData>,
    interaction_q: Query<&Interaction>,
) {
    if !mouse.pressed(MouseButton::Left) { return; }
    // If any UI element is being hovered/pressed, don't paint (click is for UI).
    for i in &interaction_q {
        if matches!(i, Interaction::Hovered | Interaction::Pressed) {
            return;
        }
    }
    let Some((x, y)) = cursor.0 else { return; };
    match *active {
        ActiveTool::None => {}
        ActiveTool::Zone(tag) => {
            if matches!(grid.get(x, y), Some(Tile::Empty)) {
                let ztype = tag.as_zone();
                let b_size = match ztype {
                    ZoneType::Residential => 3,
                    _ => 4,
                };

                // Only allow zoning if there is an adjacent road.
                let mut adjacent_road = None;
                'check: for dy in 0..b_size {
                    for dx in 0..b_size {
                        let tx = x + dx;
                        let ty = y + dy;
                        if tx >= grid.width || ty >= grid.height { continue; }
                        for (nx, ny) in grid.neighbours4(tx, ty) {
                            if let Some(Tile::Road(seg_id)) = grid.get(nx, ny) {
                                adjacent_road = Some((seg_id, (nx, ny)));
                                break 'check;
                            }
                        }
                    }
                }

                if let Some((seg_id, road_tile)) = adjacent_road {
                    let road_t = roads.get_tile_t(seg_id, road_tile);
                    spawn_building(&mut buildings, &mut grid, (x, y), ztype, seg_id, road_t);
                    grid.set_changed();
                }
            }
        }
        ActiveTool::Road | ActiveTool::Highway2x4 | ActiveTool::Highway2x8 => {
            let rtype = match *active {
                ActiveTool::Highway2x4 => RoadType::Highway2x4,
                ActiveTool::Highway2x8 => RoadType::Highway2x8,
                _ => RoadType::Normal,
            };
            if matches!(grid.get(x, y), Some(Tile::Empty) | Some(Tile::Zone(_))) {
                // Create a road link between this tile and any adjacent road tile of SAME TYPE.
                for (nx, ny) in grid.neighbours4(x, y) {
                    if let Some(Tile::Road(sid)) = grid.get(nx, ny) {
                        if roads.segments[sid as usize].road_type == rtype {
                            roads.push_link((x, y), (nx, ny), rtype);
                        }
                    }
                }
                // If no neighbours, still add as a self-link or just mark as road tile later.
                roads.push_link((x, y), (x, y), rtype);
                
                let tile_to_seg = roads.rebuild_topology();
                
                // Update the grid with the new super-segment IDs
                for ((tx, ty), seg_id) in tile_to_seg {
                    let rtype = roads.segments[seg_id as usize].road_type;
                    match rtype {
                        RoadType::Highway2x8 => {
                            for dx in -4..=5 {
                                for dy in -4..=5 {
                                    let nx = tx as i32 + dx;
                                    let ny = ty as i32 + dy;
                                    if nx >= 0 && ny >= 0 && (nx as u32) < grid.width && (ny as u32) < grid.height {
                                        grid.set(nx as u32, ny as u32, Tile::Road(seg_id));
                                    }
                                }
                            }
                        }
                        RoadType::Highway2x4 => {
                            for dx in -2..=3 {
                                for dy in -2..=3 {
                                    let nx = tx as i32 + dx;
                                    let ny = ty as i32 + dy;
                                    if nx >= 0 && ny >= 0 && (nx as u32) < grid.width && (ny as u32) < grid.height {
                                        grid.set(nx as u32, ny as u32, Tile::Road(seg_id));
                                    }
                                }
                            }
                        }
                        RoadType::Normal => {
                            for dx in 0..=1 {
                                for dy in 0..=1 {
                                    let nx = tx as u32 + dx;
                                    let ny = ty as u32 + dy;
                                    if nx < grid.width && ny < grid.height {
                                        grid.set(nx, ny, Tile::Road(seg_id));
                                    }
                                }
                            }
                        }
                    }
                    
                    // Materialize adjacent zoned tiles as buildings.
                    // Check a radius large enough for 4x4.
                    let search_radius = match rtype {
                        RoadType::Highway2x8 => 12,
                        RoadType::Highway2x4 => 8,
                        RoadType::Normal => 4,
                    };
                    for ox in -search_radius..=search_radius {
                        for oy in -search_radius..=search_radius {
                            let nx = tx as i32 + ox;
                            let ny = ty as i32 + oy;
                            if nx >= 0 && ny >= 0 && (nx as u32) < grid.width && (ny as u32) < grid.height {
                                if let Some(Tile::Zone(z)) = grid.get(nx as u32, ny as u32) {
                                    let road_tile = (tx, ty);
                                    let road_t = roads.get_tile_t(seg_id, road_tile);
                                    spawn_building(&mut buildings, &mut grid, (nx as u32, ny as u32), z, seg_id, road_t);
                                }
                            }
                        }
                    }
                }
                grid.set_changed();
            }
        }
    }
}
