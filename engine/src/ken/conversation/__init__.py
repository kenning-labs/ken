"""The Conversation — the interface-agnostic, synced log. (ARCHITECTURE.md)

Holds the outbox (renderables) and the inbox (actions) for one thread. Channels
attach and keep their own read cursor. TODO(engine): SQLite-backed queues."""
