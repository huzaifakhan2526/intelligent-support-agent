"""
WebSocket Server for Intelligent Customer Support Agent
Provides real-time chat interface over WebSocket connections
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Set, Optional
import websockets
from websockets.server import WebSocketServerProtocol

from chatbot import ChainOfThoughtChatbot

# ============================================================================
# WEBSOCKET SERVER CONFIGURATION
# ============================================================================

class WebSocketChatbotServer:
    """WebSocket server for chatbot interactions"""
    
    def __init__(self, config_path: str = "configs/config.yaml", host: str = "localhost", port: int = 8765):
        self.config_path = config_path
        self.host = host
        self.port = port
        
        # Initialize the chatbot
        self.chatbot = ChainOfThoughtChatbot(config_path)
        
        # Track active connections
        self.active_connections: Set[WebSocketServerProtocol] = set()
        self.user_sessions: Dict[str, str] = {}  # websocket_id -> user_id
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.chatbot.config.bot_name}_WebSocket")
        
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle individual WebSocket connections"""
        connection_id = str(uuid.uuid4())
        user_id = f"user_{connection_id[:8]}"
        
        # Add to active connections
        self.active_connections.add(websocket)
        self.user_sessions[connection_id] = user_id
        
        self.logger.info(f"New connection: {connection_id} (User: {user_id})")
        
        # Send welcome message
        welcome_message = {
            "type": "system",
            "message": f"Welcome to {self.chatbot.config.bot_name}! You are connected as {user_id}",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "bot_name": self.chatbot.config.bot_name,
            "version": self.chatbot.config.version
        }
        await websocket.send(json.dumps(welcome_message))
        
        try:
            # Handle incoming messages
            async for message in websocket:
                await self.handle_message(websocket, connection_id, user_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed: {connection_id}")
        except Exception as e:
            self.logger.error(f"Error handling connection {connection_id}: {str(e)}")
            error_message = {
                "type": "error",
                "message": "An error occurred while processing your message",
                "timestamp": datetime.now().isoformat()
            }
            try:
                await websocket.send(json.dumps(error_message))
            except:
                pass
        finally:
            # Clean up connection
            self.active_connections.discard(websocket)
            if connection_id in self.user_sessions:
                del self.user_sessions[connection_id]
            self.logger.info(f"Connection cleaned up: {connection_id}")
    
    async def handle_message(self, websocket: WebSocketServerProtocol, connection_id: str, user_id: str, message: str):
        """Handle incoming WebSocket messages"""
        try:
            # Parse message
            data = json.loads(message)
            message_type = data.get("type", "chat")
            content = data.get("message", "")
            session_id = data.get("session_id", "default")
            
            self.logger.info(f"Received {message_type} message from {user_id}: {content[:50]}...")
            
            if message_type == "chat":
                # Process chat message
                await self.handle_chat_message(websocket, user_id, session_id, content)
            elif message_type == "command":
                # Handle special commands
                await self.handle_command(websocket, user_id, session_id, content)
            elif message_type == "ping":
                # Respond to ping
                await websocket.send(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            else:
                # Unknown message type
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                    "timestamp": datetime.now().isoformat()
                }))
                
        except json.JSONDecodeError:
            # Handle plain text messages
            await self.handle_chat_message(websocket, user_id, "default", message)
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Failed to process message",
                "timestamp": datetime.now().isoformat()
            }))
    
    async def handle_chat_message(self, websocket: WebSocketServerProtocol, user_id: str, session_id: str, message: str):
        """Handle chat messages from users"""
        try:
            # First, send user message confirmation
            user_message = {
                "type": "chat",
                "message": message,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }
            await websocket.send(json.dumps(user_message))
            
            # Send typing indicator
            typing_message = {
                "type": "typing",
                "user_id": "bot",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(typing_message))
            
            # Get response from chatbot
            response = self.chatbot.chat(message, user_id, session_id)
            
            # Send bot response
            bot_message = {
                "type": "chat",
                "message": response,
                "user_id": "bot",
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }
            await websocket.send(json.dumps(bot_message))
            
        except Exception as e:
            self.logger.error(f"Error in chat processing: {str(e)}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Sorry, I'm having trouble processing your message right now.",
                "timestamp": datetime.now().isoformat()
            }))
    
    async def handle_command(self, websocket: WebSocketServerProtocol, user_id: str, session_id: str, command: str):
        """Handle special commands"""
        command = command.lower().strip()
        
        if command == "stats":
            # Get bot statistics
            stats = self.chatbot.get_stats()
            response = {
                "type": "system",
                "message": f"Bot Statistics: {json.dumps(stats, indent=2)}",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
            
        elif command == "clear":
            # Clear conversation history
            self.chatbot.clear_conversation(user_id, session_id)
            response = {
                "type": "system",
                "message": "Conversation history cleared!",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
            
        elif command == "history":
            # Get conversation history
            history = self.chatbot.get_conversation_history(user_id, session_id)
            response = {
                "type": "history",
                "messages": history,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
            
        else:
            # Unknown command
            response = {
                "type": "system",
                "message": f"Unknown command: {command}. Available commands: stats, clear, history",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
    
    async def broadcast_message(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        message_str = json.dumps(message)
        await asyncio.gather(
            *[ws.send(message_str) for ws in self.active_connections],
            return_exceptions=True
        )
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        self.logger.info(f"Chatbot: {self.chatbot.config.bot_name} v{self.chatbot.config.version}")
        
        # Start WebSocket server with proper handler
        async def handler(websocket):
            await self.handle_connection(websocket, "/")
        
        async with websockets.serve(handler, self.host, self.port):
            self.logger.info(f"WebSocket server is running on ws://{self.host}:{self.port}")
            self.logger.info("Press Ctrl+C to stop the server")
            
            # Keep server running
            await asyncio.Future()  # Run forever

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point for WebSocket server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebSocket Chatbot Server")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Create and start server
    server = WebSocketChatbotServer(args.config, args.host, args.port)
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 