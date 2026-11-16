# Build your own Ken dashboard

Ken Cloud has a web app. A self-hosted Ken does not. But the web app uses only the public MCP tools of the
engine, so you can build the same thing for your own Ken. Point Claude (or any coding agent) at this file and
your engine's MCP URL, and ask it to build the page.

This file lists what the page needs. Every call is one MCP `tools/call` to your engine
(`http://localhost:<port>/mcp` with your user token as `Authorization: Bearer …`).

## The parts

| Part | Read | Act |
|---|---|---|
| **Chat** with Ken | `chat_history {source, limit}` | `tell {text, source, provenance: "user"}` |
| **To confirm**: writes that wait for your yes | `pending {}` | `decide {card_id, approved}` |
| **Overnight**: what the Dreamer found (needs `dreamer.enabled`) | `inbox {status: "all", readable: true}` | `inbox_chat`, `inbox_apply`, `inbox_close` |
| **Today**: your tasks and your day | `tasks {}` | `complete_task`, `schedule_task`, `reschedule_task`, `discard_task` |

Use one fixed `source` for your chat (for example `"My dashboard"`). The thread is stored in the engine, so it is
the same on every device.

## Show what the engine gives you

The engine writes the words for you. Show them as they are. Do not build your own wording from the ids.

- **A card** (`pending`) has `show`:
  `{question, yes, no, done: [if yes, if no]}`. For example:
  "Update what Ken knows about Michael?", with "Yes, update" and "No, keep it". After the answer, show
  `done[0]` or `done[1]` ("Updated Michael").
  A card that changes a value has `existing` (now) and `question` (new). Show both, one above the other.
  `evidence` says where the card came from. Show it folded ("Based on 2 notes").
- **An overnight item** (`inbox`, with `readable: true`) has task names in place of ids, and dates in plain words
  ("13 Nov, 17:45"). Its `kind` says what it is:
  - `ask`: it needs you. Show `title`, `body` and `quick` (suggested answers; the first one is Ken's pick).
    Send the tapped answer, or the user's own words, with `inbox_chat {item_id, text}`. Ken carries it out and
    closes the item when it is decided.
  - `look`: worth knowing, no decision. Same shape, and a Dismiss button (`inbox_close {item_id, outcome}`).
  - `tidy`: a fact for a person's or project's record. It has `show` like a card (`question`, `yes`, `no`,
    `done: [if yes, if no]`) and `add` (the text to add, exactly as it will be written).
    Yes is `inbox_apply {item_id}`. If the user changed the wording, send their version:
    `inbox_apply {item_id, add: "…"}`. Skip is `inbox_close {item_id, outcome: "skipped"}`.
  - A closed item (`status: "closed"`) has `outcome`. Show it under "Answered".

Without `readable: true` you get the ids, for an agent that acts on the tasks. A page wants `readable: true`.

## Read the answer of every write

Every write returns `{outcome, text, …}`. Show `text`.

- `queued`: Ken has it and does it soon. It is not a failure. **Never send it again**: that makes a duplicate.
- `forbidden`, `error`, `busy`, `blocked`, `not_saved`, `no_item`, `no_such_card`: nothing happened. Show it as a
  failure, and read the data again.
- `reasked` or `restaled` (from `decide`): Ken raised a new card in place of this one. Read `pending` again.

After each answer, read `pending` and `inbox` again. The engine is the truth; do not keep your own copy.

## Good to have

These are client behaviour. Ken Cloud has them, and your page can have them too:

- **Undo**: show the answer as done at once, and send it after 5 seconds unless the user taps Undo. Send it at
  once when the page closes (`pagehide`).
- **Add all** for the tidy items: apply them one at a time, and count the ones that fail.
- **One receipt for many changes**: `inbox_chat` returns `actions`. Fold more than three into one line.
