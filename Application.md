# Application

### Model Context Protocol (MCP)

MCP is an open protocol introduced by Anthropic that standardizes how AI models interact with external tools, data, and services.

Instead of manually injecting tool definitions into system prompts, MCP provides a unified way for models to discover and use tools through providers.

How it works:

- Developers configure MCP providers (local or remote).
- The MCP runtime automatically fetches available tools, their input/output schemas, and safely injects them into the model’s context.
- The model can then call these tools as if they were built-in functions.

Difference from HTTP:

- HTTP is a transport protocol (how to send/receive bytes).
- MCP is a semantic protocol (how models understand and use external capabilities).
- HTTP moves data, MCP explains capabilities. MCP turns external APIs and resources into a standardized “toolbox” that AI models can understand and safely use.