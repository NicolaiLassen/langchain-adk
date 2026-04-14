"""Smoke test for pyink (pyinklib) chat UI."""
from pyink import Box, Text, component, render
from pyink.hooks import use_animation, use_app, use_input, use_state


@component
def chat():
    messages, set_messages = use_state(["Welcome to orx!", "Type /help, q to quit"])
    buf, set_buf = use_state("")
    app = use_app()

    def on_key(ch, key):
        if key.ctrl and ch == "d":
            app.exit()
            return
        if key.return_key:
            msg = buf.strip()
            if msg:
                set_messages(lambda m: [*m, f"> {msg}", f"  Echo: {msg}"])
                set_buf("")
            return
        if key.backspace or key.delete:
            set_buf(lambda t: t[:-1] if t else t)
            return
        if ch and not key.ctrl and not key.meta and not key.escape:
            set_buf(lambda t: t + ch)

    use_input(on_key)

    return Box(
        *[Text(m) for m in messages],
        Text(f"orx> {buf}\u2588", color="#6C8EBF", bold=True),
        flex_direction="column",
    )


if __name__ == "__main__":
    render(chat())
