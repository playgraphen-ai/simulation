use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy::render::extract_resource::ExtractResource;

use crate::sim::buildings::BuildingData;
use crate::sim::grid::{CityGrid, Tile, ZoneType};
use crate::sim::people::{Activity, PeopleData};
use crate::sim::roads::RoadData;
use crate::ui::tools::ActiveTool;
use crate::render::camera::CursorTile;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectedObj {
    Building(u32),
    Person(u32),
    Zone(ZoneType),
    Road(u32),
}

#[derive(Resource, Default, Clone, ExtractResource)]
pub struct Selection {
    pub obj: Option<SelectedObj>,
    pub changed_frame: u32,
}

#[derive(Component)]
pub struct InspectorPanel;

#[derive(Component)]
pub struct InspectorText;

#[derive(Component)]
pub struct InspectorToggleBtn;

#[derive(Component)]
pub struct InspectorContent;

pub fn setup_inspector(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(12.),
            bottom: Val::Px(12.),
            width: Val::Px(280.),
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(Val::Px(8.)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.1, 0.1, 0.15, 0.9)),
        InspectorPanel,
    )).with_children(|p| {
        // Top bar with toggle button
        p.spawn((
            Node {
                width: Val::Percent(100.0),
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                margin: UiRect::bottom(Val::Px(8.)),
                ..default()
            },
        )).with_children(|top_bar| {
            top_bar.spawn((
                Text::new("INSPECTOR"),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::WHITE),
            ));
            
            top_bar.spawn((
                Button,
                Node {
                    padding: UiRect::horizontal(Val::Px(8.)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.3, 0.3, 0.3, 1.0)),
                InspectorToggleBtn,
            )).with_children(|btn| {
                btn.spawn((
                    Text::new("Hide"),
                    TextFont { font_size: 12.0, ..default() },
                    TextColor(Color::WHITE),
                ));
            });
        });

        // Content
        p.spawn((
            Node {
                flex_direction: FlexDirection::Column,
                display: Display::Flex,
                ..default()
            },
            InspectorContent,
        )).with_children(|content| {
            content.spawn((
                Text::new("Select 'None (camera)'\nClick on a car or building."),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::WHITE),
                InspectorText,
            ));
        });
    });
}

pub fn toggle_inspector(
    mut interaction_query: Query<(&Interaction, &Children), (Changed<Interaction>, With<InspectorToggleBtn>)>,
    mut text_query: Query<&mut Text>,
    mut content_query: Query<&mut Node, With<InspectorContent>>,
) {
    for (interaction, children) in &mut interaction_query {
        if *interaction == Interaction::Pressed {
            if let Ok(mut content_node) = content_query.single_mut() {
                let is_hidden = content_node.display == Display::None;
                
                if is_hidden {
                    content_node.display = Display::Flex;
                    if let Ok(mut text) = text_query.get_mut(children[0]) {
                        text.0 = "Hide".to_string();
                    }
                } else {
                    content_node.display = Display::None;
                    if let Ok(mut text) = text_query.get_mut(children[0]) {
                        text.0 = "Show".to_string();
                    }
                }
            }
        }
    }
}

pub fn handle_selection(
    mouse: Res<ButtonInput<MouseButton>>,
    active: Res<ActiveTool>,
    cursor: Res<CursorTile>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    grid: Res<CityGrid>,
    people: Res<PeopleData>,
    roads: Res<RoadData>,
    mut selection: ResMut<Selection>,
    interaction_q: Query<&Interaction, With<Button>>,
) {
    if !mouse.just_pressed(MouseButton::Left) { return; }
    if !matches!(*active, ActiveTool::None) { return; }
    
    for i in &interaction_q {
        if matches!(i, Interaction::Hovered | Interaction::Pressed) {
            return; // Clicked on UI button
        }
    }

    // 1. Try to select a car via raycast
    let mut closest_car = None;
    let mut min_dist = 4.0; // Click radius for cars (increased for easier clicking)

    if let Ok(window) = windows.single() {
        if let Some(pos) = window.cursor_position() {
            if let Ok((camera, cam_tf)) = camera_q.single() {
                if let Ok(ray) = camera.viewport_to_world(cam_tf, pos) {
                    let dir: Vec3 = ray.direction.into();
                    for id in 0..people.len {
                        let row = &people.rows[id as usize];
                        if row.activity_code as u32 == Activity::Travelling as u32 && row.current_seg != 0xFFFFFFFFu32 as f32 {
                            let seg_id = row.current_seg as u32;
                            if let Some(seg) = roads.segments.get(seg_id as usize) {
                                let cx = (seg.a.0 + seg.b.0) as f32 * 0.5 + 0.5;
                                let cz = (seg.a.1 + seg.b.1) as f32 * 0.5 + 0.5;
                                let c_pos = Vec3::new(cx, 0.0, cz);
                                
                                let dist = (c_pos - ray.origin).cross(dir).length();
                                if dist < min_dist {
                                    min_dist = dist;
                                    closest_car = Some(id);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(car_id) = closest_car {
        selection.obj = Some(SelectedObj::Person(car_id));
        selection.changed_frame += 1;
        return;
    }

    // 2. Try to select ground object using the existing Camera CursorTile
    if let Some((x, z)) = cursor.0 {
        match grid.get(x, z) {
            Some(Tile::Building(bid)) => {
                selection.obj = Some(SelectedObj::Building(bid));
                selection.changed_frame += 1;
                return;
            }
            Some(Tile::Zone(ztype)) => {
                selection.obj = Some(SelectedObj::Zone(ztype));
                selection.changed_frame += 1;
                return;
            }
            Some(Tile::Road(rid)) => {
                selection.obj = Some(SelectedObj::Road(rid));
                selection.changed_frame += 1;
                return;
            }
            _ => {}
        }
    }

    if selection.obj.is_some() {
        selection.changed_frame += 1;
    }
    selection.obj = None;
}

pub fn update_inspector_ui(
    selection: Res<Selection>,
    buildings: Res<BuildingData>,
    people: Res<PeopleData>,
    roads: Res<RoadData>,
    mut text_q: Query<&mut Text, With<InspectorText>>,
) {
    let Ok(mut text) = text_q.single_mut() else { return; };

    match selection.obj {
        None => {
            text.0 = "INSPECTOR\n\nSelect 'None (camera)'\nClick on a car, building, or zone.".to_string();
        }
        Some(SelectedObj::Zone(ztype)) => {
            text.0 = format!(
                "--- ZONE ---\nType: {:?}\nStatus: Empty\n\nZones need an adjacent road\nto spawn a building.",
                ztype
            );
        }
        Some(SelectedObj::Road(rid)) => {
            if let Some(seg) = roads.segments.get(rid as usize) {
                text.0 = format!(
                    "--- ROAD ---\nID: {}\nSpeed multiplier: {:.2}x\nLength: {:.1}m\nConnections: A({}), B({})",
                    rid, seg.speed_mean, seg.length, seg.conn_a.len(), seg.conn_b.len()
                );
            }
        }
        Some(SelectedObj::Building(bid)) => {
            if let Some(b) = buildings.items.get(bid as usize) {
                let status = if b.capacity == 0 { "ABANDONED" } else { "ACTIVE" };
                text.0 = format!(
                    "--- BUILDING {} ---\nStatus: {}\nType: {:?}\nLevel: {}\nOccupants: {} (Phys)\nAssigned: {} / {}\nIncome: {:.1}\nGrowth: {:.3}\nAge: {:.0}s",
                    bid, status, b.btype, b.level, b.occupants, b.assigned, b.capacity, b.income, b.growth, b.age_seconds
                );
            } else {
                text.0 = "Building destroyed".to_string();
            }
        }
        Some(SelectedObj::Person(pid)) => {
            if (pid as usize) < people.rows.len() {
                let r = &people.rows[pid as usize];
                let act = match r.activity_code as u32 {
                    0 => "Travelling",
                    1 => "At Home",
                    2 => "At Work",
                    3 => "Shopping",
                    _ => "Unknown",
                };
                let h_id = if r.home as u32 == 0xFFFFFFFF { "None".to_string() } else { (r.home as u32).to_string() };
                let w_id = if r.work as u32 == 0xFFFFFFFF { "None".to_string() } else { (r.work as u32).to_string() };
                let d_id = if r.destination as u32 == 0xFFFFFFFF { "None".to_string() } else { (r.destination as u32).to_string() };
                
                text.0 = format!(
                    "--- PERSON {} ---\nAge: {:.0}\nMoney: ${:.0}\nHome ID: {}\nWork ID: {}\nDest ID: {}\nActivity: {}\nTime left: {:.1}s",
                    pid, r.age, r.money, h_id, w_id, d_id, act, r.activity_time
                );
            }
        }
    }
}
