"""
HTTP Server for serving the web interface
Runs alongside the WebSocket server to provide a complete web experience
"""

import asyncio
import logging
from pathlib import Path
from aiohttp import web, WSMsgType
from aiohttp.web import Request, Response
import json
from datetime import datetime

from websocket_server import WebSocketChatbotServer

# ============================================================================
# HTTP SERVER CONFIGURATION
# ============================================================================

class HTTPChatbotServer:
    """HTTP server for serving web interface and handling WebSocket upgrades"""
    
    def __init__(self, config_path: str = "configs/config.yaml", host: str = "localhost", http_port: int = 8080, ws_port: int = 8765):
        self.config_path = config_path
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        
        # Initialize the chatbot
        self.chatbot = WebSocketChatbotServer(config_path, host, ws_port)
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.chatbot.chatbot.config.bot_name}_HTTP")
        
        # Create aiohttp app
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP routes"""
        # Serve static files
        static_path = Path(__file__).parent.parent / "static"
        self.app.router.add_static('/static', static_path)
        
        # Main page
        self.app.router.add_get('/', self.handle_index)
        
        # Health check
        self.app.router.add_get('/health', self.handle_health)
        
        # API endpoints
        self.app.router.add_get('/api/stats', self.handle_api_stats)
        self.app.router.add_post('/api/chat', self.handle_api_chat)
    
    async def handle_index(self, request: Request) -> Response:
        """Serve the main chat interface"""
        static_path = Path(__file__).parent.parent / "static"
        index_file = static_path / "index.html"
        
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return web.Response(text=content, content_type='text/html')
        else:
            return web.Response(text="Chat interface not found", status=404)
    
    async def handle_health(self, request: Request) -> Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "bot_name": self.chatbot.chatbot.config.bot_name,
            "version": self.chatbot.chatbot.config.version,
            "timestamp": datetime.now().isoformat()
        })
    
    async def handle_api_stats(self, request: Request) -> Response:
        """API endpoint for bot statistics"""
        try:
            stats = self.chatbot.chatbot.get_stats()
            return web.json_response(stats)
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}")
            return web.json_response({"error": "Failed to get statistics"}, status=500)
    
    async def handle_api_chat(self, request: Request) -> Response:
        """API endpoint for chat (fallback for non-WebSocket clients)"""
        try:
            data = await request.json()
            message = data.get('message', '')
            user_id = data.get('user_id', 'api_user')
            session_id = data.get('session_id', 'api_session')
            
            if not message:
                return web.json_response({"error": "Message is required"}, status=400)
            
            # Get response from chatbot
            response = self.chatbot.chatbot.chat(message, user_id, session_id)
            
            return web.json_response({
                "response": response,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error in API chat: {str(e)}")
            return web.json_response({"error": "Failed to process message"}, status=500)
    
    async def start_server(self):
        """Start the HTTP server"""
        self.logger.info(f"Starting HTTP server on {self.host}:{self.http_port}")
        self.logger.info(f"WebSocket server will run on {self.host}:{self.ws_port}")
        self.logger.info(f"Chatbot: {self.chatbot.chatbot.config.bot_name} v{self.chatbot.chatbot.config.version}")
        
        # Start HTTP server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.http_port)
        await site.start()
        
        self.logger.info(f"HTTP server is running on http://{self.host}:{self.http_port}")
        self.logger.info(f"WebSocket server is running on ws://{self.host}:{self.ws_port}")
        self.logger.info("Press Ctrl+C to stop the server")
        
        # Keep server running
        await asyncio.Future()  # Run forever

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point for HTTP server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HTTP Chatbot Server")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP port to bind to")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port to bind to")
    
    args = parser.parse_args()
    
    # Create and start server
    server = HTTPChatbotServer(args.config, args.host, args.http_port, args.ws_port)
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 