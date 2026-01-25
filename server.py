import socket
import threading
import sys
from src.main import (
    warmup_model,
    create_user_session,
    process_message
)

HOST = "0.0.0.0"
# PORT = 5001
clients = []
clients_lock = threading.Lock()
running = True
warmup_model()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to an external host (doesn't actually send data)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def handle_client(conn, addr, args):
    with clients_lock:
        clients.append(conn)
    print(f"[CONNECTED] {addr}")

    # log_probs_eval = initialize()

    try:
        conn.send(b"Please enter your user ID: ")
        user_id_data = conn.recv(1024)
        if not user_id_data:
            raise Exception("No user ID received")

        try:
            user_id = int(user_id_data.decode().strip())
        except ValueError:
            conn.send(b"Invalid user ID. Disconnecting.")
            conn.close()
            return

        retrievers, router = create_user_session(args, user_id)
        conversation = []
        filtered_convo = []

        conn.send(b"Login successful. You can start chatting now.\n\nYou: ")

        while True:
            data = conn.recv(4096)
            if not data:
                break

            user_input = data.decode().strip()
            if user_input.lower() in {"exit", "quit"}:
                conn.send(b"SERVER_SHUTDOWN")
                break
            reply = process_message(user_id, user_input, args, conversation, filtered_convo, retrievers, router)
            # prompt, reply = generate_slm_result(user_id, user_input, args, conversation, filtered_convo, user_data)

            # perform_post_slm_computation(conversation, filtered_convo, log_probs_eval, prompt, reply)
            conn.send(reply.encode())

    except Exception as e:
        print(f"[ERROR] Client {addr}: {e}")
    finally:
        with clients_lock:
            if conn in clients:
                clients.remove(conn)
        conn.close()
        print(f"[DISCONNECTED] {addr}")


def shutdown_server():
    """Notify all clients and close connections."""
    global running
    print("[SERVER] Shutting down...")
    running = False

    with clients_lock:
        for conn in clients[:]:
            try:
                conn.send(b"SERVER_SHUTDOWN")
                conn.close()
            except:
                pass
        clients.clear()


def server_console():
    """Thread to monitor 'exit' command."""
    while True:
        cmd = input().strip().lower()
        if cmd == "exit":
            shutdown_server()
            break


def start_server(args):
    global running
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, args.port))
    server_socket.listen()
    print(f"[SERVER] Listening on {get_local_ip()}:{args.port}")
    print("Type 'exit' to shut down the server.\n")

    threading.Thread(target=server_console, daemon=True).start()

    try:
        while running:
            try:
                server_socket.settimeout(1.0)
                conn, addr = server_socket.accept()
                threading.Thread(target=handle_client, args=(conn, addr, args), daemon=True).start()
            except socket.timeout:
                continue
    finally:
        server_socket.close()
        print("[SERVER] Server stopped.")
        sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--port", "-p", default=5001, type=int)
    parser.add_argument("--generate_data", "-gd", action="store_true")
    args = parser.parse_args()

    start_server(args)
