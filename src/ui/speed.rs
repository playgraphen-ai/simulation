use bevy::prelude::*;

#[derive(Component)]
pub struct SpeedRoot;

#[derive(Component, Clone, Copy, PartialEq)]
pub enum SpeedAction {
    Pause,
    Play1x,
    Play2x,
    Play10x,
}

pub fn setup_speed_ui(mut commands: Commands) {
    // We use a full-width transparent container to center the actual panel.
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.),
            width: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            ..default()
        },
        SpeedRoot,
        bevy::ui::FocusPolicy::Pass,
    )).with_children(|root| {
        root.spawn((
            Node {
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(10.),
                padding: UiRect::all(Val::Px(8.)),
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.1, 0.1, 0.15, 0.85)),
            bevy::ui::FocusPolicy::Block,
        )).with_children(|panel| {
            spawn_speed_btn(panel, "|| Pause", SpeedAction::Pause);
            spawn_speed_btn(panel, "> x1", SpeedAction::Play1x);
            spawn_speed_btn(panel, ">> x2", SpeedAction::Play2x);
            spawn_speed_btn(panel, ">>> x10", SpeedAction::Play10x);
        });
    });
}

fn spawn_speed_btn(p: &mut ChildSpawnerCommands, text: &str, action: SpeedAction) {
    p.spawn((
        Button,
        Node {
            width: Val::Px(80.0),
            height: Val::Px(30.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(Color::srgb(0.2, 0.2, 0.25)),
        action,
    )).with_children(|b| {
        b.spawn((
            Text::new(text),
            TextFont { font_size: 16.0, ..default() },
            TextColor(Color::WHITE),
        ));
    });
}

pub fn handle_speed_buttons(
    mut q: Query<(&Interaction, &SpeedAction), Changed<Interaction>>,
    mut time: ResMut<Time<Virtual>>,
) {
    for (interaction, action) in &mut q {
        if *interaction == Interaction::Pressed {
            match action {
                SpeedAction::Pause => { time.pause(); },
                SpeedAction::Play1x => { time.unpause(); time.set_relative_speed(1.0); },
                SpeedAction::Play2x => { time.unpause(); time.set_relative_speed(2.0); },
                SpeedAction::Play10x => { time.unpause(); time.set_relative_speed(10.0); },
            }
        }
    }
}

pub fn update_speed_button_colors(
    time: Res<Time<Virtual>>,
    mut buttons: Query<(&SpeedAction, &Interaction, &mut BackgroundColor)>,
) {
    let is_paused = time.is_paused();
    let speed = time.relative_speed();
    
    for (action, interaction, mut bg) in &mut buttons {
        let active = match action {
            SpeedAction::Pause => is_paused,
            SpeedAction::Play1x => !is_paused && (speed - 1.0).abs() < f32::EPSILON,
            SpeedAction::Play2x => !is_paused && (speed - 2.0).abs() < f32::EPSILON,
            SpeedAction::Play10x => !is_paused && (speed - 10.0).abs() < f32::EPSILON,
        };
        
        let color = if active {
            Color::srgb(0.2, 0.6, 0.2) // Active green
        } else if matches!(*interaction, Interaction::Hovered) {
            Color::srgb(0.3, 0.4, 0.5) // Hover grey-blue
        } else {
            Color::srgb(0.2, 0.2, 0.25) // Inactive grey
        };
        
        // Only update if changed to avoid unnecessary re-renders
        if bg.0 != color {
            bg.0 = color;
        }
    }
}