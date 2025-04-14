import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
    # methods will go here

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        # Enhanced system message with explicit instructions about tool usage
        system_message = """You are a highly experienced researcher who carefully understands the user query, breaks it into relevant parts and uses tools to get the data for every topic.

Important guidelines:
1. Use tools only when necessary to gather specific information.
2. After each tool call, evaluate if you have sufficient information to answer the query.
3. If you have enough information, provide a comprehensive final answer without making additional tool calls.
4. Limit yourself to at most 3-4 tool calls total to avoid repetition.
5. If you notice you're calling similar tools repeatedly, stop and synthesize what you already have.
6. Always provide a clear, final conclusion that synthesizes all the information gathered."""

        # Enhance the query to encourage a final synthesis
        enhanced_query = f"""{query}

After gathering the necessary information using tools, please provide a comprehensive final answer that synthesizes all the data."""

        messages = [
            {
                "role": "user",
                "content": enhanced_query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        print("------ AVAILABLE TOOLS: ", available_tools)

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=system_message,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []
        
        # Keep track of the conversation
        conversation_messages = messages.copy()
        
        # Add safeguards against infinite loops
        max_iterations = 5  # Maximum number of tool calls
        iteration_count = 0
        used_tools = set()  # Track which tools have been used
        
        while iteration_count < max_iterations:
            iteration_count += 1
            print(f"------ ITERATION {iteration_count}/{max_iterations}")
            
            # Check if Claude's response indicates it's ready to conclude
            text_content = " ".join([c.text for c in response.content if c.type == 'text'])
            conclusion_signals = [
                "In conclusion,", 
                "To summarize,", 
                "In summary,",
                "Based on all the information gathered,",
                "Here's the complete analysis:"
            ]
            
            if any(signal in text_content for signal in conclusion_signals) and iteration_count > 1:
                print("------ DETECTED CONCLUSION SIGNAL")
                # Add the final text to our results
                final_text.append(text_content)
                break
            
            # Process the current response
            assistant_message_content = []
            has_tool_use = False
            
            for content in response.content:
                if content.type == 'text':
                    final_text.append(content.text)
                    assistant_message_content.append(content)
                elif content.type == 'tool_use':
                    has_tool_use = True
                    tool_name = content.name
                    tool_args = content.input
                    tool_use_id = content.id
                    
                    # Check if we're repeatedly calling the same tool with the same args
                    import json
                    tool_call_signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                    if tool_call_signature in used_tools:
                        print(f"------ WARNING: Repeated tool call detected: {tool_call_signature}")
                        # We can either break the loop or continue with a warning
                        final_text.append(f"[Notice: Repeated tool call to {tool_name} detected. Skipping to prevent loop.]")
                        # Force the loop to end after this iteration
                        has_tool_use = False
                        break
                    
                    used_tools.add(tool_call_signature)

                    print("------ CALLING TOOL: ", tool_name, tool_args)
                    # Execute tool call
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                        print("------ TOOL RESULT: ", result)
                        
                        # Add assistant's message with tool use to conversation
                        conversation_messages.append({
                            "role": "assistant",
                            "content": response.content
                        })
                        
                        # Add tool result to conversation
                        conversation_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": result.content
                                }
                            ]
                        })
                    except Exception as e:
                        print(f"------ TOOL ERROR: {str(e)}")
                        final_text.append(f"[Error calling tool {tool_name}: {str(e)}]")
                        # Add error as tool result
                        conversation_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": f"Error: {str(e)}"
                                }
                            ]
                        })
            
            # If no tool was used or we've processed all tool calls, break the loop
            if not has_tool_use:
                # Add the final assistant message to conversation
                conversation_messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                break
                
            # Get next response from Claude with updated conversation
            try:
                print("------ SENDING UPDATED CONVERSATION TO CLAUDE")
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=system_message,
                    messages=conversation_messages,
                    tools=available_tools
                )
            except Exception as e:
                print(f"------ CLAUDE API ERROR: {str(e)}")
                final_text.append(f"\nError getting response from Claude: {str(e)}")
                break
        
        # Check if we hit the iteration limit
        if iteration_count >= max_iterations:
            final_text.append("\n[Notice: Maximum number of tool calls reached. The analysis may be incomplete.]")
            
            # Make one final call to Claude to synthesize what we have so far
            try:
                conversation_messages.append({
                    "role": "user",
                    "content": "We've reached the maximum number of tool calls. Please provide a final summary based on the information gathered so far."
                })
                
                final_response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=system_message,
                    messages=conversation_messages
                )
                
                final_text.append(final_response.content[0].text)
            except Exception as e:
                print(f"------ FINAL SYNTHESIS ERROR: {str(e)}")
        
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())

