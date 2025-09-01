import smtplib
import ssl
from email.message import EmailMessage
import subprocess
import yaml
import os
import json
import argparse

def load_config():
    # Load email configuration from a YAML file in the home directory (see example_email_config.yaml)
    config_path = os.path.join(os.path.expanduser('~'), 'email_config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['email']
    except Exception as e:
        print(f"Error loading config file: {e}")
        raise

def get_json_content(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Pretty print JSON with indentation
            return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading JSON file: {e}"

def get_nvidia_smi_output():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error getting nvidia-smi output: {e}"

def get_disk_usage():
    try:
        # Get full df output
        result = subprocess.run(['df', '-h'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        header = lines[0]
        
        # Look for root filesystem, handling different possible mount points
        root_lines = [line for line in lines[1:] if line.strip() and (
            line.split()[-1] == '/' or  # Standard root
            '/dev/root' in line or      # Some systems use /dev/root
            'rootfs' in line            # Some systems show as rootfs
        )]
        
        if not root_lines:
            return "Could not find root filesystem information"
            
        root_info = root_lines[0]
        return f"{header}\n{root_info}"
    except Exception as e:
        return f"Error getting disk usage: {e}"

def get_python_processes():
    try:
        # Get all processes containing "python", with full command
        result = subprocess.run(['ps', '-eo', 'command'], 
                              capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        # Filter for python processes and extract just the command
        python_processes = []
        for line in lines[1:]:  # Skip header
            if 'python' in line.lower():
                # Clean up the command line
                cmd = line.strip()
                if cmd.startswith('python'):  # Remove 'python' prefix if present
                    cmd = cmd[6:].strip()
                if cmd.startswith('python3'):  # Remove 'python3' prefix if present
                    cmd = cmd[7:].strip()
                python_processes.append(cmd)
        
        if not python_processes:
            return "No Python processes currently running", "No Python processes currently running"
        
        # Create HTML version with bullet points
        html_output = f"""<div style="font-family: Arial, sans-serif;">
<h4>Running Python Scripts:</h4>
<ul style="margin: 0; padding-left: 20px;">
{''.join(f'<li style="margin-bottom: 5px;">{cmd}</li>' for cmd in python_processes)}
</ul>
</div>"""

        # Create plain text version
        plain_output = "Running Python Scripts:\n" + "\n".join(f"- {cmd}" for cmd in python_processes)
        
        return html_output, plain_output
    except Exception as e:
        error_msg = f"Error getting Python processes: {e}"
        return error_msg, error_msg

def send_email(json_path=None):
    # Load email configuration
    config = load_config()
    
    # Default subject
    subject = "System Status Update"
    
    # Get JSON content if path provided
    json_content = ""
    if json_path:
        json_content = get_json_content(json_path)
        json_filename = os.path.basename(json_path)
        try:
            # Try to parse JSON to get save_folder
            json_data = json.loads(json_content)
            if 'save_folder' in json_data:
                subject = f"Run Execute d- {json_data['save_folder']}"
        except:
            pass  # If JSON parsing fails, use default subject
    
    nvidia_output = get_nvidia_smi_output()
    disk_usage = get_disk_usage()
    
    # Get process info
    html_processes, plain_processes = get_python_processes()
    
    # Create HTML version of the email
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h3>GPU Status (nvidia-smi output):</h3>
        <pre style="background-color: #f5f5f5; 
                    padding: 15px; 
                    border-radius: 5px; 
                    font-family: 'Courier New', monospace; 
                    white-space: pre; 
                    overflow-x: auto;
                    font-size: 12px;
                    border: 1px solid #ddd;">{nvidia_output}</pre>

        <h3>Running Python Processes:</h3>
        <div style="background-color: #f5f5f5; 
                    padding: 15px; 
                    border-radius: 5px;
                    font-size: 12px;
                    border: 1px solid #ddd;
                    overflow-x: auto;">{html_processes}</div>

        <h3>Disk Usage Summary (df -h):</h3>
        <pre style="background-color: #f5f5f5; 
                    padding: 15px; 
                    border-radius: 5px; 
                    font-family: 'Courier New', monospace; 
                    white-space: pre; 
                    overflow-x: auto;
                    font-size: 12px;
                    border: 1px solid #ddd;">{disk_usage}</pre>"""

    # Add JSON content section if provided
    if json_path:
        html_body += f"""
        <h3>JSON File Contents ({json_filename}):</h3>
        <pre style="background-color: #f5f5f5; 
                    padding: 15px; 
                    border-radius: 5px; 
                    font-family: 'Courier New', monospace; 
                    white-space: pre; 
                    overflow-x: auto;
                    font-size: 12px;
                    border: 1px solid #ddd;">{json_content}</pre>"""

    html_body += """
    </body>
    </html>
    """
    
    # Create plain text version as fallback
    plain_body = f"""System Status Update

GPU Status (nvidia-smi output):
{nvidia_output}

Running Python Processes:
{plain_processes}

Disk Usage Summary (df -h):
{disk_usage}"""

    # Add JSON content to plain text version if provided
    if json_path:
        plain_body += f"""

JSON File Contents ({json_filename}):
{json_content}"""

    msg = EmailMessage()
    msg.set_content(plain_body)  # Plain text version
    msg.add_alternative(html_body, subtype='html')  # HTML version
    msg["Subject"] = subject
    msg["From"] = config['sender_email']
    msg["To"] = config['receiver_email']

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL(config['smtp_server'], config['smtp_port'], context=context) as server:
            server.login(config['sender_email'], config['sender_password'])
            server.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Run your script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send system status email with optional JSON file contents')
    parser.add_argument('--json', type=str, help='Path to JSON file to include in email')
    args = parser.parse_args()
    
    print("Sending system status email...")
    send_email(args.json)
