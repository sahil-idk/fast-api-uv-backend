import os
from dotenv import load_dotenv
import json
import asyncio
import logging
import argparse
from typing import Dict, List, Optional, Any
import uuid
import traceback

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from websockets.exceptions import ConnectionClosed

# Load environment variables from .env file
load_dotenv()

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log loaded environment variables (masked for security)
api_key = os.getenv("ULTRAVOX_API_KEY")
if api_key:
    masked_key = f"{api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else '***'}"
    logger.info(f"Loaded ULTRAVOX_API_KEY from environment: {masked_key}")
else:
    logger.warning("No ULTRAVOX_API_KEY found in environment variables!")

# Create FastAPI app
app = FastAPI(title="Ultravox Web Bridge")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; in production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage of active sessions
active_sessions: Dict[str, Dict[str, Any]] = {}

class CallRequest(BaseModel):
    system_prompt: Optional[str] = Field(None, description="System prompt for the AI assistant")
    voice: Optional[str] = Field(None, description="Voice ID to use")
    temperature: Optional[float] = Field(0.8, description="Temperature for response generation")
    user_speaks_first: Optional[bool] = Field(True, description="Whether user speaks first")
    api_key: Optional[str] = Field(None, description="Optional API key override")

# Add middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        raise

async def create_ultravox_call(call_params: CallRequest) -> Dict[str, Any]:
    """Create a new Ultravox call and return the join URL and session ID."""
    api_key = call_params.api_key or os.getenv('ULTRAVOX_API_KEY')
    
    # Enhanced API key logging
    if not api_key:
        logger.error("ULTRAVOX_API_KEY not found in environment variables or request")
        raise HTTPException(status_code=400, detail="ULTRAVOX_API_KEY not provided")
    else:
        # Safely log partial key for debugging
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        logger.info(f"Using API key: {masked_key}")
    
    target = "https://api.ultravox.ai/api/calls"
    logger.info(f"Making request to: {target}")
    
    async with aiohttp.ClientSession() as session:
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        system_prompt = call_params.system_prompt or """
        You are an AI assistant. The user is talking to you over voice on their phone, 
        and your response will be read out loud with realistic text-to-speech (TTS) technology.
        
        Please be concise, clear, and helpful. Use natural conversational language.
        """
        
        body = {
            "systemPrompt": system_prompt,
            "temperature": call_params.temperature,
            "medium": {
                "serverWebSocket": {
                    "inputSampleRate": 16000,  # Changed from 48000 to 16000
                    "outputSampleRate": 16000, # Changed from 48000 to 16000
                    "clientBufferSizeMs": 30000,
                }
            }
        }
        
        if call_params.voice:
            body["voice"] = call_params.voice
            
        if call_params.user_speaks_first is not None:
            body["firstSpeaker"] = "FIRST_SPEAKER_USER" if call_params.user_speaks_first else "FIRST_SPEAKER_AGENT"
        
        logger.info(f"Creating call with body: {json.dumps(body, indent=2)}")
        
        try:
            async with session.post(target, headers=headers, json=body) as response:
                status_code = response.status
                response_text = await response.text()
                
                logger.info(f"Response status code: {status_code}")
                
                # Try to log JSON response if possible
                try:
                    response_data = json.loads(response_text)
                    logger.info(f"Response data: {json.dumps(response_data, indent=2)}")
                except json.JSONDecodeError:
                    logger.info(f"Response text (not JSON): {response_text}")
                
                # Accept both 200 and 201 as success codes
                if status_code not in [200, 201]:
                    logger.error(f"Error response from Ultravox API: {status_code} - {response_text}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Failed to create call: {status_code}, {response_text}"
                    )
                
                # Try to parse the response as JSON
                try:
                    response_json = json.loads(response_text)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse Ultravox API response as JSON: {response_text}")
                    raise HTTPException(status_code=500, detail="Invalid response from Ultravox API")
                
                if "joinUrl" not in response_json:
                    logger.error(f"Ultravox API response missing joinUrl: {response_json}")
                    raise HTTPException(status_code=500, detail="Invalid response from Ultravox API (missing joinUrl)")
                    
                join_url = response_json["joinUrl"]
                call_id = response_json.get("callId", "unknown")
                logger.info(f"Received join URL: {join_url}")
                logger.info(f"Call ID: {call_id}")
                
                # Add API version parameter to the join URL
                if "?" in join_url:
                    join_url += "&apiVersion=1"
                else:
                    join_url += "?apiVersion=1"
                
                session_id = str(uuid.uuid4())
                logger.info(f"Generated session ID: {session_id}")
                
                return {
                    "session_id": session_id,
                    "call_id": call_id,
                    "join_url": join_url
                }
        except aiohttp.ClientError as e:
            logger.error(f"Client error in Ultravox API request: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Client error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error creating Ultravox call: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to create call: {str(e)}")

@app.post("/api/create-call")
async def create_call(call_request: CallRequest):
    """Create a new call and return a session ID."""
    try:
        # Log the incoming request
        logger.info(f"Received create-call request: {call_request.dict()}")
        
        result = await create_ultravox_call(call_request)
        # Store the session info without connecting yet
        active_sessions[result["session_id"]] = {
            "join_url": result["join_url"],
            "call_id": result["call_id"],
            "ultravox_socket": None,
            "client_sockets": [],
            "state": "created",
            "transcript": "",
            "is_speaking": False
        }
        
        logger.info(f"Call created successfully with session ID: {result['session_id']}")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in create_call: {e}")
        logger.error(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint to handle realtime communication with client and Ultravox."""
    client_id = str(uuid.uuid4())[:8]  # Generate a short client ID for logging
    logger.info(f"WebSocket connection request for session {session_id} (client {client_id})")
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted for client {client_id}")
        
        # Validate session
        if session_id not in active_sessions:
            logger.error(f"Invalid session ID: {session_id}")
            await websocket.send_json({
                "type": "error", 
                "message": "Invalid session ID",
                "details": f"Session {session_id} not found in active sessions"
            })
            await websocket.close(code=4003)  # Custom close code for invalid session
            return
        
        session = active_sessions[session_id]
        logger.info(f"Session found: {session_id}, current state: {session.get('state', 'unknown')}")
        
        # Add this client to the session's connected clients
        session["client_sockets"].append(websocket)
        
        # Establish Ultravox connection if not already established
        if not session.get("ultravox_socket"):
            logger.info(f"Initiating Ultravox connection for session {session_id}")
            try:
                from websockets.client import connect
                ultravox_socket = await connect(session['join_url'])
                session["ultravox_socket"] = ultravox_socket
                session["state"] = "connected"
                logger.info(f"Successfully connected to Ultravox for session {session_id}")
                
                # Start task to handle messages from Ultravox
                asyncio.create_task(handle_ultravox_messages(session_id, ultravox_socket))
            except Exception as connect_error:
                logger.error(f"Failed to connect to Ultravox: {connect_error}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to establish Ultravox connection",
                    "details": str(connect_error)
                })
                session["client_sockets"].remove(websocket)
                await websocket.close(code=4004)
                return
        
        # Handle messages from client
        try:
            while True:
                message = await websocket.receive()
                
                if message["type"] == "websocket.disconnect":
                    logger.info(f"Client {client_id} disconnected")
                    break
                
                if "bytes" in message:
                    # Audio data from client - forward to Ultravox
                    if session.get("ultravox_socket"):
                        await session["ultravox_socket"].send(message["bytes"])
                    else:
                        logger.warning(f"Cannot forward audio: no active Ultravox connection for session {session_id}")
                        
                elif "text" in message:
                    # Control message from client
                    try:
                        data = json.loads(message["text"])
                        logger.debug(f"Received control message from client: {data}")
                        
                        # Handle client control messages here if needed
                        if data.get("type") == "mute_mic":
                            logger.info(f"Client {client_id} muted microphone")
                        elif data.get("type") == "unmute_mic":
                            logger.info(f"Client {client_id} unmuted microphone")
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON message from client: {message['text']}")
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for client {client_id}")
        except Exception as e:
            logger.error(f"Error handling client messages: {e}")
            logger.error(traceback.format_exc())
        finally:
            # Remove this client from the session
            if websocket in session["client_sockets"]:
                session["client_sockets"].remove(websocket)
                logger.info(f"Removed client {client_id} from session {session_id}")
            
            # If no clients are left, close the ultravox connection
            if not session["client_sockets"] and session.get("ultravox_socket"):
                logger.info(f"No clients left. Closing Ultravox connection for session {session_id}")
                await session["ultravox_socket"].close()
                session["ultravox_socket"] = None
                session["state"] = "disconnected"
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}")
        logger.error(traceback.format_exc())

async def handle_ultravox_messages(session_id: str, ultravox_socket):
    """Process messages coming from Ultravox and distribute to all connected clients."""
    session = active_sessions[session_id]
    
    try:
        logger.info(f"Started Ultravox message handler for session {session_id}")
        async for message in ultravox_socket:
            if isinstance(message, bytes):
                # Audio from Ultravox - forward to all clients
                for client_ws in session["client_sockets"]:
                    try:
                        await client_ws.send_bytes(message)
                    except Exception as e:
                        logger.error(f"Error sending audio to client: {e}")
                        # Client will be removed in the main handler when it disconnects
            else:
                # Control message from Ultravox
                try:
                    msg_data = json.loads(message)
                    logger.debug(f"Received control message from Ultravox: {msg_data}")
                    
                    # Process message based on type
                    if msg_data.get("type") == "state":
                        old_state = session.get("state", "unknown")
                        new_state = msg_data.get("state")
                        session["state"] = new_state
                        session["is_speaking"] = new_state == "speaking"
                        
                        logger.info(f"Session {session_id} state changed: {old_state} -> {new_state}")
                        
                        # Forward state change to all clients
                        for client_ws in session["client_sockets"]:
                            try:
                                await client_ws.send_json({
                                    "type": "state",
                                    "state": new_state,
                                    "is_speaking": session["is_speaking"]
                                })
                            except Exception as e:
                                logger.error(f"Error sending state to client: {e}")
                    
                    elif msg_data.get("type") == "transcript" and msg_data.get("role") == "agent":
                        # Process agent transcript
                        if msg_data.get("text"):
                            session["transcript"] = msg_data["text"]
                            logger.info(f"Session {session_id} transcript set: {msg_data['text'][:50]}...")
                        elif msg_data.get("delta"):
                            session["transcript"] += msg_data.get("delta", "")
                            if len(msg_data.get("delta", "")) > 0:
                                logger.debug(f"Session {session_id} transcript delta: {msg_data.get('delta')}")
                        
                        # Forward transcript to all clients
                        for client_ws in session["client_sockets"]:
                            try:
                                await client_ws.send_json({
                                    "type": "transcript",
                                    "text": session["transcript"],
                                    "final": msg_data.get("final", False)
                                })
                            except Exception as e:
                                logger.error(f"Error sending transcript to client: {e}")
                    
                    elif msg_data.get("type") == "playback_clear_buffer":
                        logger.info(f"Session {session_id}: Clearing audio buffer")
                        # Forward to all clients
                        for client_ws in session["client_sockets"]:
                            try:
                                await client_ws.send_json({
                                    "type": "clear_buffer"
                                })
                            except Exception as e:
                                logger.error(f"Error sending clear_buffer to client: {e}")
                    
                    # Forward all message types to clients for debugging/monitoring
                    for client_ws in session["client_sockets"]:
                        try:
                            await client_ws.send_json({
                                "type": "ultravox_message",
                                "data": msg_data
                            })
                        except Exception as e:
                            logger.error(f"Error forwarding message to client: {e}")
                
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse message from Ultravox: {message}")
                except Exception as e:
                    logger.error(f"Error processing Ultravox message: {e}")
                    logger.error(traceback.format_exc())
    
    except ConnectionClosed:
        logger.info(f"Ultravox connection closed for session {session_id}")
    except Exception as e:
        logger.error(f"Error in Ultravox message handler: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Update session state
        session["state"] = "disconnected"
        if session.get("ultravox_socket"):
            try:
                await session["ultravox_socket"].close()
            except:
                pass
            session["ultravox_socket"] = None
        
        logger.info(f"Ultravox message handler ended for session {session_id}")

@app.get("/api/session/{session_id}")
async def get_session_status(session_id: str):
    """Get status information about a session."""
    logger.info(f"Requested status for session {session_id}")
    
    if session_id not in active_sessions:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    response = {
        "state": session["state"],
        "is_speaking": session["is_speaking"],
        "client_count": len(session["client_sockets"]),
        "call_id": session.get("call_id", "unknown")
    }
    logger.info(f"Returning status for session {session_id}: {response}")
    return response

@app.delete("/api/session/{session_id}")
async def end_session(session_id: str):
    """End an active session."""
    logger.info(f"Request to end session {session_id}")
    
    if session_id not in active_sessions:
        logger.warning(f"Session not found for deletion: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Close Ultravox connection
    if session["ultravox_socket"]:
        logger.info(f"Closing Ultravox connection for session {session_id}")
        await session["ultravox_socket"].close()
    
    # Close all client connections
    logger.info(f"Closing {len(session['client_sockets'])} client connections for session {session_id}")
    for client_ws in session["client_sockets"]:
        await client_ws.close()
    
    # Remove session from active sessions
    del active_sessions[session_id]
    logger.info(f"Removed session {session_id} from active sessions")
    
    return {"status": "success"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "active_sessions": len(active_sessions)}

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Ultravox Web Bridge")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server to")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        # Set debug logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)