import platform

from pynput import keyboard


SYSTEM = platform.system()
WALK_STRENGTH = 0.55
SPRINT_STRENGTH = 1.0

if SYSTEM == "Windows":
    import pyvjoy

    vjoy_device = pyvjoy.VJoyDevice(1)

    def _to_vjoy(value: float) -> int:
        value = max(-1.0, min(1.0, value))
        return max(1, min(0x8000, round((value + 1.0) * 0x4000)))

    def set_stick(x: float, y: float) -> None:
        vjoy_device.set_axis(pyvjoy.HID_USAGE_X, _to_vjoy(x))
        vjoy_device.set_axis(pyvjoy.HID_USAGE_Y, _to_vjoy(y))

    def set_button(button: int, pressed: bool) -> None:
        vjoy_device.set_button(button, int(pressed))

    def close_joystick() -> None:
        vjoy_device.reset()

elif SYSTEM == "Linux":
    from evdev import AbsInfo, UInput, ecodes

    axis = AbsInfo(
        value=0,
        min=-32768,
        max=32767,
        fuzz=0,
        flat=0,
        resolution=0,
    )
    capabilities = {
        ecodes.EV_KEY: [ecodes.BTN_JOYSTICK, ecodes.BTN_TRIGGER],
        ecodes.EV_ABS: [
            (ecodes.ABS_X, axis),
            (ecodes.ABS_Y, axis),
        ],
    }
    uinput_device = UInput(
        capabilities,
        name="Python Virtual Joystick",
        version=1,
    )
    buttons = {
        1: ecodes.BTN_JOYSTICK,
        2: ecodes.BTN_TRIGGER,
    }

    def set_stick(x: float, y: float) -> None:
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        uinput_device.write(ecodes.EV_ABS, ecodes.ABS_X, round(x * 32767))
        uinput_device.write(ecodes.EV_ABS, ecodes.ABS_Y, round(y * 32767))
        uinput_device.syn()

    def set_button(button: int, pressed: bool) -> None:
        if button not in buttons:
            raise ValueError(f"Unsupported button: {button}")
        uinput_device.write(ecodes.EV_KEY, buttons[button], int(pressed))
        uinput_device.syn()

    def close_joystick() -> None:
        uinput_device.close()

else:
    raise RuntimeError(f"Unsupported operating system: {SYSTEM}")


def move_left(strength: float = WALK_STRENGTH) -> None:
    set_stick(-strength, 0.0)


def move_right(strength: float = WALK_STRENGTH) -> None:
    set_stick(strength, 0.0)


def move_up(strength: float = WALK_STRENGTH) -> None:
    set_stick(0.0, -strength)


def move_down(strength: float = WALK_STRENGTH) -> None:
    set_stick(0.0, strength)


def stop_moving() -> None:
    set_stick(0.0, 0.0)


direction_keys = {
    keyboard.Key.left: (-1.0, 0.0),
    keyboard.Key.right: (1.0, 0.0),
    keyboard.Key.up: (0.0, -1.0),
    keyboard.Key.down: (0.0, 1.0),
}
pressed_directions = set()
pressed_modifiers = set()
button_pressed = False
listener: keyboard.Listener | None = None


def hotkey_active() -> bool:
    ctrl_pressed = bool(
        {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
        & pressed_modifiers
    )
    alt_pressed = bool(
        {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr}
        & pressed_modifiers
    )
    return ctrl_pressed and alt_pressed


def sprint_active() -> bool:
    return bool(
        {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r}
        & pressed_modifiers
    )


def update_movement() -> None:
    if not hotkey_active():
        stop_moving()
        return

    strength = SPRINT_STRENGTH if sprint_active() else WALK_STRENGTH
    x = sum(direction_keys[key][0] for key in pressed_directions) * strength
    y = sum(direction_keys[key][1] for key in pressed_directions) * strength
    set_stick(max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y)))


def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
    global button_pressed

    if key is None:
        return

    if key in {
        keyboard.Key.ctrl,
        keyboard.Key.ctrl_l,
        keyboard.Key.ctrl_r,
        keyboard.Key.alt,
        keyboard.Key.alt_l,
        keyboard.Key.alt_r,
        keyboard.Key.alt_gr,
        keyboard.Key.shift,
        keyboard.Key.shift_l,
        keyboard.Key.shift_r,
    }:
        pressed_modifiers.add(key)
        update_movement()

    if key in direction_keys:
        pressed_directions.add(key)
        update_movement()
    elif isinstance(key, keyboard.KeyCode) and key.char:
        character = key.char.lower()
        if hotkey_active() and character == "j" and not button_pressed:
            button_pressed = True
            set_button(1, True)
        elif hotkey_active() and character == "q":
            if listener is not None:
                listener.stop()


def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
    global button_pressed

    if key is None:
        return

    pressed_modifiers.discard(key)
    if key in direction_keys:
        pressed_directions.discard(key)
    if isinstance(key, keyboard.KeyCode) and key.char:
        if key.char.lower() == "j" and button_pressed:
            button_pressed = False
            set_button(1, False)
    update_movement()


print("Global controls:")
print("  Ctrl+Alt+Arrow keys: move (hold multiple arrows for diagonals)")
print("  Shift: sprint while moving")
print("  Ctrl+Alt+J: joystick button 1")
print("  Ctrl+Alt+Q: quit")

try:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
finally:
    stop_moving()
    set_button(1, False)
    close_joystick()
