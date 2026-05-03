//! Toolbar with the four build tools, the spawn button and activity sliders.

use bevy::prelude::*;

use crate::sim::grid::ZoneType;
use crate::sim::{ActivityDurations, SimSettings, SpawnPeopleRequest};

#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTool {
    #[default]
    None,
    Road,
    Zone(ZoneTag),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoneTag {
    Residential,
    Office,
    Shop,
}

impl ZoneTag {
    pub fn as_zone(self) -> ZoneType {
        match self {
            ZoneTag::Residential => ZoneType::Residential,
            ZoneTag::Office => ZoneType::Office,
            ZoneTag::Shop => ZoneType::Shop,
        }
    }
}

#[derive(Component, Clone, Copy)]
pub struct ToolButton(pub ActiveTool);

#[derive(Component)]
pub struct SpawnButton;

/// Tagged sliders. Drag-click on them adjusts the corresponding value.
#[derive(Component, Clone, Copy, Debug)]
pub enum SliderKind {
    HomeDuration,
    WorkDuration,
    ShopDuration,
    HomeToWorkProb,
    AbandonMultiplier,
    RentCost,
    WorkSalary,
    ShopCost,
}

#[derive(Component)]
pub struct SliderFill(pub SliderKind);

pub fn setup_toolbar(mut commands: Commands) {
    // Root panel, right-hand side.
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(12.),
            top: Val::Px(12.),
            width: Val::Px(220.),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(8.),
            padding: UiRect::all(Val::Px(10.)),
            ..default()
        },
        BackgroundColor(Color::srgba(0., 0., 0., 0.55)),
    ))
    .with_children(|p| {
        label(p, "BUILD");
        tool_button(p, "Road", ActiveTool::Road);
        tool_button(p, "Zone: Residential", ActiveTool::Zone(ZoneTag::Residential));
        tool_button(p, "Zone: Office", ActiveTool::Zone(ZoneTag::Office));
        tool_button(p, "Zone: Shop", ActiveTool::Zone(ZoneTag::Shop));
        tool_button(p, "None (camera)", ActiveTool::None);

        label(p, "POPULATION");
        spawn_button(p);

        label(p, "ECONOMY & RULES");
        slider_row(p, "Abandon x", SliderKind::AbandonMultiplier, 0.1, 5.0, 1.0);
        slider_row(p, "Rent Cost", SliderKind::RentCost, 0.0, 100.0, 20.0);
        slider_row(p, "Work Pay", SliderKind::WorkSalary, 10.0, 200.0, 50.0);
        slider_row(p, "Shop Cost", SliderKind::ShopCost, 0.0, 100.0, 30.0);

        label(p, "ACTIVITIES (s)");
        slider_row(p, "Home", SliderKind::HomeDuration, 0.0, 120.0, 30.0);
        slider_row(p, "Work", SliderKind::WorkDuration, 0.0, 120.0, 45.0);
        slider_row(p, "Shop", SliderKind::ShopDuration, 0.0, 120.0, 15.0);
        slider_row(p, "P(home→work)", SliderKind::HomeToWorkProb, 0.0, 1.0, 0.6);
    });
}

fn label(p: &mut ChildSpawnerCommands, s: &str) {
    p.spawn((
        Text::new(s),
        TextFont { font_size: 14.0, ..default() },
        TextColor(Color::srgb(0.9, 0.9, 0.9)),
        Node { margin: UiRect::top(Val::Px(6.)), ..default() },
    ));
}

fn tool_button(p: &mut ChildSpawnerCommands, s: &str, tool: ActiveTool) {
    p.spawn((
        Button,
        Node {
            width: Val::Percent(100.),
            height: Val::Px(28.),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(Color::srgba(0.2, 0.2, 0.25, 1.)),
        ToolButton(tool),
    ))
    .with_children(|b| {
        b.spawn((
            Text::new(s),
            TextFont { font_size: 13.0, ..default() },
            TextColor(Color::WHITE),
        ));
    });
}

fn spawn_button(p: &mut ChildSpawnerCommands) {
    p.spawn((
        Button,
        Node {
            width: Val::Percent(100.),
            height: Val::Px(30.),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(Color::srgba(0.25, 0.5, 0.25, 1.)),
        SpawnButton,
    ))
    .with_children(|b| {
        b.spawn((
            Text::new("+10 people"),
            TextFont { font_size: 13.0, ..default() },
            TextColor(Color::WHITE),
        ));
    });
}

fn slider_row(p: &mut ChildSpawnerCommands, label_s: &str, kind: SliderKind, min: f32, max: f32, init: f32) {
    p.spawn((
        Node {
            width: Val::Percent(100.),
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(6.),
            align_items: AlignItems::Center,
            ..default()
        },
    )).with_children(|row| {
        row.spawn((
            Text::new(label_s),
            TextFont { font_size: 12.0, ..default() },
            TextColor(Color::srgb(0.85, 0.85, 0.85)),
            Node { width: Val::Px(80.), ..default() },
        ));
        row.spawn((
            Node {
                width: Val::Percent(100.),
                height: Val::Px(14.),
                ..default()
            },
            BackgroundColor(Color::srgba(0.12, 0.12, 0.15, 1.)),
            Interaction::default(),
            bevy::ui::RelativeCursorPosition::default(),
            kind,
            SliderRange { min, max },
        )).with_children(|track| {
            let ratio = ((init - min) / (max - min)).clamp(0., 1.);
            track.spawn((
                Node {
                    width: Val::Percent(ratio * 100.),
                    height: Val::Percent(100.),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.25, 0.6, 0.9, 1.)),
                SliderFill(kind),
                bevy::ui::FocusPolicy::Pass,
            ));
        });
    });
}

#[derive(Component, Clone, Copy, Debug)]
pub struct SliderRange { pub min: f32, pub max: f32 }

pub fn handle_tool_buttons(
    mut active: ResMut<ActiveTool>,
    mut interactions: Query<
        (&Interaction, &ToolButton, &mut BackgroundColor),
        (Changed<Interaction>, With<Button>),
    >,
    all_buttons: Query<(&ToolButton, &mut BackgroundColor), Without<Interaction>>,
) {
    let _ = all_buttons; // reserved if we want to re-tint siblings later
    for (interaction, btn, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                *active = btn.0;
                bg.0 = Color::srgba(0.4, 0.6, 0.8, 1.);
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.3, 0.3, 0.35, 1.);
            }
            Interaction::None => {
                bg.0 = if *active == btn.0 {
                    Color::srgba(0.4, 0.6, 0.8, 1.)
                } else {
                    Color::srgba(0.2, 0.2, 0.25, 1.)
                };
            }
        }
    }
}

pub fn handle_spawn_button(
    mut q: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<SpawnButton>)>,
    mut ev: MessageWriter<SpawnPeopleRequest>,
) {
    for (interaction, mut bg) in &mut q {
        match *interaction {
            Interaction::Pressed => {
                ev.write(SpawnPeopleRequest { count: 10 });
                bg.0 = Color::srgba(0.4, 0.7, 0.4, 1.);
            }
            Interaction::Hovered => bg.0 = Color::srgba(0.3, 0.55, 0.3, 1.),
            Interaction::None => bg.0 = Color::srgba(0.25, 0.5, 0.25, 1.),
        }
    }
}

/// Reads mouse interaction on the slider track and rewrites its fill width
/// and the underlying resource value.
pub fn handle_sliders(
    mouse: Res<ButtonInput<MouseButton>>,
    mut durations: ResMut<ActivityDurations>,
    mut settings: ResMut<SimSettings>,
    mut tracks: Query<(&Interaction, &bevy::ui::RelativeCursorPosition, &SliderKind, &SliderRange)>,
    mut fills: Query<(&SliderFill, &mut Node)>,
) {
    if !mouse.pressed(MouseButton::Left) { return; }
    for (interaction, rel_cursor, kind, range) in &mut tracks {
        if !matches!(*interaction, Interaction::Hovered | Interaction::Pressed) { continue; }
        if let Some(pos) = rel_cursor.normalized {
            let ratio = pos.x.clamp(0.0, 1.0);
            let value = range.min + (range.max - range.min) * ratio;
            match kind {
                SliderKind::HomeDuration => durations.home = value,
                SliderKind::WorkDuration => durations.work = value,
                SliderKind::ShopDuration => durations.shop = value,
                SliderKind::HomeToWorkProb => durations.home_to_work_prob = value.clamp(0., 1.),
                SliderKind::AbandonMultiplier => settings.abandon_multiplier = value,
                SliderKind::RentCost => settings.rent_cost = value,
                SliderKind::WorkSalary => settings.work_salary = value,
                SliderKind::ShopCost => settings.shop_cost = value,
            }
            for (fill, mut fill_node) in &mut fills {
                if std::mem::discriminant(&fill.0) == std::mem::discriminant(kind) {
                    fill_node.width = Val::Percent(ratio * 100.);
                }
            }
        }
    }
}
