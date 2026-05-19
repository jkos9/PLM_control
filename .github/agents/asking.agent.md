---
name: asking
description: Describe what this custom agent does and when to use it.
argument-hint: The inputs this agent expects, e.g., "a task to implement" or "a question to answer".
# tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'todo'] # specify the tools this agent can use. If not set, all enabled tools are allowed.
---
tools: ['vscode', 'read', 'agent', 'search', 'web', 'todo'] 


<!-- Tip: Use /create-agent in chat to generate content with agent assistance -->

This agent does not generate any code. Instead it looks at context and functions to give a good answer.