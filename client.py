import socket
import threading
import sys
import queue

def listen_server(sock, stop_event):
    """Thread constantly listens for messages from the server."""
    while not stop_event.is_set():
        try:
            sock.settimeout(1.0)
            msg = sock.recv(1024).decode()
            if not msg:
                continue
            if msg == "SERVER_SHUTDOWN":
                print("\nServer is shutting down. Exiting...")
                stop_event.set()
                break
            print(f"\nServer: {msg}\n\nYou: ", end="", flush=True)
        except socket.timeout:
            continue
        except:
            break
    sock.close()
    sys.exit(0)

def input_thread(input_queue, stop_event):
    """Thread to read user input without blocking shutdown."""
    while not stop_event.is_set():
        try:
            msg = input().strip()
            input_queue.put(msg)
        except EOFError:
            stop_event.set()
            break

def run_client():
    SERVER_IP = input("Enter server IP: ").strip()
    PORT = int(input("Enter server port: ").strip())

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((SERVER_IP, PORT))
    except:
        print("Could not connect — server offline or wrong IP/port")
        return

    msg = client_socket.recv(1024).decode()
    print(msg)
    user_id = input("User ID: ").strip()
    client_socket.send(user_id.encode())

    login_reply = client_socket.recv(1024).decode()
    print(login_reply)

    if "Invalid" in login_reply:
        client_socket.close()
        return

    stop_event = threading.Event()
    input_queue = queue.Queue()

    threading.Thread(target=listen_server, args=(client_socket, stop_event), daemon=True).start()
    threading.Thread(target=input_thread, args=(input_queue, stop_event), daemon=True).start()

    while not stop_event.is_set():
        try:
            msg = input_queue.get(timeout=0.5)
        except:
            continue

        if msg.lower() == "exit":
            stop_event.set()
            break

        try:
            client_socket.send(msg.encode())
        except:
            stop_event.set()
            break

    client_socket.close()
    sys.exit(0)

if __name__ == "__main__":
    run_client()
