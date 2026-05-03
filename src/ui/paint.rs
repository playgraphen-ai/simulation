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
                        adjacent_road = Some(seg_id);
                        break;
                    }
                }

                if let Some(seg_id) = adjacent_road {
                    grid.set(x, y, Tile::Zone(tag.as_zone()));
                    spawn_building(&mut buildings, &mut grid, (x, y), tag.as_zone(), seg_id);
                    grid.set_changed();
                }
            }
        }
        ActiveTool::Road => {
            if matches!(grid.get(x, y), Some(Tile::Empty) | Some(Tile::Zone(_))) {
                // Create a road segment whose endpoints are this tile and its
                // closest existing road neighbour (if any). Otherwise this is
                // a 1-tile segment; it will auto-connect to adjacent roads
                // added later.
                let mut neighbour_tile = None;
                for (nx, ny) in grid.neighbours4(x, y) {
                    if let Some(Tile::Road(_)) = grid.get(nx, ny) {
                        neighbour_tile = Some((nx, ny));
                        break;
                    }
                }
                let other = neighbour_tile.unwrap_or((x, y));
                if let Some(seg_id) = roads.push_segment((x, y), other) {
                    grid.set(x, y, Tile::Road(seg_id));
                    // Materialize adjacent zoned tiles as buildings.
                    let neighbours: Vec<(u32, u32)> = grid.neighbours4(x, y).collect();
                    for (nx, ny) in neighbours {
                        if let Some(Tile::Zone(z)) = grid.get(nx, ny) {
                            spawn_building(&mut buildings, &mut grid, (nx, ny), z, seg_id);
                        }
                    }
                    grid.set_changed();
                }
            }
        }
    }
}
