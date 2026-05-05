//! "Painting" system: when a tool is active and the user left-clicks (and
//! holds) over a grid tile that is not covered by UI, stamp the tool output
//! there. For road tool, also auto-materialize buildings on adjacent zoned
//! tiles.

use bevy::prelude::*;

use crate::render::camera::CursorTile;
use crate::sim::buildings::{spawn_building, BuildingData};
use crate::sim::grid::{CityGrid, Tile};
use crate::sim::roads::RoadData;

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
                // Only allow zoning if there is an adjacent road.
                // This satisfies "fais qu'on peut zoner qu'a proximité des routes"
                // and fixes the bug where buildings only materialize if road is built after.
                let mut adjacent_road = None;
                for (nx, ny) in grid.neighbours4(x, y) {
                    if let Some(Tile::Road(seg_id)) = grid.get(nx, ny) {
                        adjacent_road = Some((seg_id, (nx, ny)));
                        break;
                    }
                }

                if let Some((seg_id, road_tile)) = adjacent_road {
                    grid.set(x, y, Tile::Zone(tag.as_zone()));
                    let road_t = roads.get_tile_t(seg_id, road_tile);
                    spawn_building(&mut buildings, &mut grid, (x, y), tag.as_zone(), seg_id, road_t);
                    grid.set_changed();
                }
            }
        }
        ActiveTool::Road => {
            if matches!(grid.get(x, y), Some(Tile::Empty) | Some(Tile::Zone(_))) {
                // Create a road link between this tile and any adjacent road tile.
                for (nx, ny) in grid.neighbours4(x, y) {
                    if let Some(Tile::Road(_)) = grid.get(nx, ny) {
                        roads.push_link((x, y), (nx, ny));
                    }
                }
                // If no neighbours, still add as a self-link or just mark as road tile later.
                roads.push_link((x, y), (x, y));
                
                let tile_to_seg = roads.rebuild_topology();
                
                // Update the grid with the new super-segment IDs
                for ((tx, ty), seg_id) in tile_to_seg {
                    // Mark a 2x2 area around each road point to reserve space for the wider road.
                    for dx in 0..2 {
                        for dy in 0..2 {
                            if tx + dx < grid.width && ty + dy < grid.height {
                                grid.set(tx + dx, ty + dy, Tile::Road(seg_id));
                            }
                        }
                    }
                    
                    // Materialize adjacent zoned tiles as buildings.
                    // We check a slightly larger radius for buildings now.
                    for ox in -1..=2i32 {
                        for oy in -1..=2i32 {
                            let nx = tx as i32 + ox;
                            let ny = ty as i32 + oy;
                            if nx >= 0 && ny >= 0 && (nx as u32) < grid.width && (ny as u32) < grid.height {
                                if let Some(Tile::Zone(z)) = grid.get(nx as u32, ny as u32) {
                                    let road_t = roads.get_tile_t(seg_id, (tx, ty));
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
