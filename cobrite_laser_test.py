# Felix Correll - 27.01.25
from datetime import datetime as d_time
try:
    import serial
except:
    pass
import socket
from time import sleep

##############################################################################################################################
# Configuration
##############################################################################################################################
isConfigured = True #set to True after the configurations below; otherwise the demo script won't run completely.
isUsbConnection = True # True = use USB; False = use ethernet/ip
ipAddress = "192.168.0.1" # this is the ip address of the unit
port = 2000 # do NOT change this number; only used when is UsbConneciton = False
usbComPort = "COM10" # this sets the com port of the device; for details check the manual; only used when isUsbConnection = True 
"""
Set to:
    - 'ABC/OMFT' for ABC or OMFT device
    - 'DX1' for DX1 device
    - 'Cobrite without DX1' for DX2,DX,MX or CORX device (MX: please power up the laser cards before running the script)
"""
deviceType = "DX1" 

#####################################################################################################
"""
README:

1) Installation
-----------------------------------------------------------------------------------------------------
    1.1)
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    optional, but highly recommended:
    create a virtual environment
    ```
    python -m venv .venv
    .\.venv\Script\Activate.ps1     (powershell)
    source ./.venv/activate         (bash)
    ```
    1.2)
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        1.2.1)
        If connection via ethernet no requirements are necessary!
        You can skip this step.
        1.2.2)
        If connection via usb:
        ```
        pip install -r requirements.txt
        ```
2) Usage
-----------------------------------------------------------------------------------------------------
    2.1)
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        2.1.1)
        First of all select wether to use usb or ehternet via the switch in the configurations section. 
        Also define the ip address.
        If your device is a laser ensure that 'isLaser' is set to True. Otherwise it should be set to False.
        If your device is a DX1 'isDX1' should be set to True. Otherwise it should be set to False. 
        2.1.2)
        In dependence of your device (isLaser and isDX1) commands from commands_laser, commands_no_laser and commands_dx1 are executed.
        You can find these three list below in the section called 'List of commands to be executed'.
        If you want a new command to be executed just add it to a list that is executed.
        There are three possibilities:
            1) isLaser = True and isDX1 = False : commands from commands_laser are executed
            2) isLaser = True and isDX1 = True : commands from commands_dx1 are executed
            3) isLaser = False (isDX1 doesn't matter) : commands from commands_no_laser are executed
        2.1.3)
        You can pause the program for x seconds by sending 'sleep x'; x has to be a natural number.
        For documentation of the commands please check the manual of your device!
"""


##############################################################################################################################
# System Class
##############################################################################################################################
# class to communicate with idp device
class System():
    def __init__(self, host="192.168.0.1",port=2000,usb_com_port="COM20",usb=False):
        self.host = host
        self.port = port
        self.usbComPort = usb_com_port
        self.usb = usb
        self.timeout = 100

    # sending Scpi command to device via ehternet or usb
    def sendScpi(self,command):
        reply = ""
        # connection via usb
        if self.usb:
            sepChar = ';'
            self.serial.write((command + sepChar).encode('utf-8'))
            reply = ""
            start_time = d_time.now()
            while reply.find(sepChar) < 0:
                reply = reply + self.serial.read(255).decode("utf-8")
                if(d_time.now() - start_time).seconds > self.timeout:
                    print("ERROR: SCPI timed out")
                    break
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            self.serial.write((command + sepChar).encode('utf-8'))
            reply = ""
            start_time = d_time.now()
            while reply.find(sepChar) < 0:
                reply = reply + self.serial.read(255).decode("utf-8")
                if (d_time.now() - start_time).seconds > self.timeout:
                    print("ERROR: SCPI timed out")
                    break
            if "ERR" in reply:
                raise(Exception(f'command: {command} -> reply: {reply}'))
        # connection via ethernet
        else:
            self.socket.sendall(bytearray(command + "\n", 'utf-8'))
            start_time = d_time.now()
            while reply.find(';') < 0:
                reply = reply + self.socket.recv(1024).decode("utf-8")
                if (d_time.now() - start_time).seconds > self.timeout:
                    print("ERROR: SCPI timed out")
                    break
            if "ERR" in reply:
                raise(Exception(f'command: {command} -> reply: {reply}'))
        # create output and return reply
        print(f'command: {command} -> reply: {reply}')
        return reply
    
    # create connection via usb or ethernet
    def createConnection(self):
        try:
            if self.usb:
                t_out = 0
                try:
                    serial.Serial(self.usbComPort, 115200, timeout=t_out).close()
                except:
                    pass
                self.serial = serial.Serial(self.usbComPort, 115200, timeout=t_out)
                self.conntype = self.serial
            else:
                self.socket = socket.socket()
                self.socket.connect((self.host, self.port))
                self.conntype = self.socket
        except Exception as e:
            raise(e)
        

#####################################################################################################
# List of commands to be executed
#####################################################################################################
# the following three lists are exectued in dependence of the deviceType variable in the Configuration section above
# used for DX2,DX,MX and CORX
commands_cobrite_without_dx1 = ["*idn?","pass IDP","lay?","stat 1,1,1,1","bwai 1,1,1","conf? 1,1,1","sleep 15","wav? 1,1,1","freq? 1,1,1","freq?","freq? 1,1,*","pow? 1,1,1","stat 1,1,1,0"]
# used for ABC and OMFT
commands_abc = ["*idn?","pass IDP","serno?","macaddress?","partno?"]
# used for OMFT
commands_omft = ["lay?","stat 1,1,1,1","sleep 15","conf? 1,1,1","wav? 1,1,1","freq? 1,1,1","freq?","freq? 1,1,*","pow? 1,1,1","stat 1,1,1,0"]
# used for DX1
commands_dx1 = ["*idn?","pass IDP","intl?","mon?","busy?","lay?","stat 1,1,1,1","conf? 1,1,1","wav? 1,1,1","freq? 1,1,1","freq?","pow? 1,1,1","stat 1,1,1,0"]
 # these commands are used to set values; please use the manual to ensure that these values are compatible with your device!
 # feel free to include them in the commands_laser list above to test them
commands_to_try_for_yourself = ["freq 1,1,1,191.5","pow 1,1,1,9.5","wav 1,1,1,1570"]

"""
Overview of used commands:
- *idn? : queries idn string of unit
- pass IDP: sets user level to 1
- lay? : queries chassis configuration
- stat 1,1,1,1 : enables laser 1,1,1
- stat 1,1,1,0 : disables laser 1,1,1
- bwai 1,1,1 : unit will acknowledge once laser 1,1,1 has finished tuning (use 'bwai' to wait until all laser ports have finished tuning)
- conf? 1,1,1 : queries current configuration of laser in location 1,1,1
- wav? 1,1,1 : queries wavelength of laser in location 1,1,1
- freq? 1,1,1 : queries frequency of laser in location 1,1,1
- freq? : queries frequency of laser in location 1,1,1 (if no further information given default laser is always 1,1,1)
- freq? 1,1,* : queries frequency of all lasers on lasercard 1 (check user manual for more detailed information about the usage of *)
- pow? 1,1,1 : queries power of laser in location 1,1,1
- serno? : queries serialnumber of device
- intl? : queries interlock state of device
- mon?: queries monitor readings from laser (<LD chip Temperature>, format nn.nn, unit °C <LD base Temperature>, format nnnn.n, unit mA <LD chip current>, format nnnn.n, unit mA <TEC current>, format nnnn.n, unit mA) 
- busy? : queries if laser port is currently tuned “1” or settled “0”.
- freq 1,1,1,191.5 : sets frequency of laser at location 1,1,1 to 191.5 THz
- pow 1,1,1,9.5 : sets power of laser at location 1,1,1 to 9.5 dBm 
- wav 1,1,1,1570 : sets wavelength of laser at location 1,1,1 to 1570 nm

For more detailed information please check the user manual
"""
#####################################################################################################
def run_demo():
    try:
        if not isConfigured:
            print("="*100)
            print("Please check the configuration section before you run the script for the first time.\nYou can find it on top of this python file"+
            " below of the imports.\nThere you have to set your device type, if you are using an ethernet or usb connection,\nwhich ip address or usb port you are using\nand finally you should"+
            " set the isConfigured variable to True.")  
            print("="*100)
            return 
        # creating System instance
        idp_device = System(host=ipAddress,port=port,usb_com_port=usbComPort,usb=isUsbConnection)
        # establish connection
        idp_device.createConnection()
        if "cobrite" in deviceType.lower():
            commands = commands_cobrite_without_dx1
        elif "dx1" in deviceType.lower():
            commands = commands_dx1
        elif "abc" in deviceType.lower() or "omft" in deviceType.lower():
            commands = commands_abc
            if "omft" in idp_device.sendScpi("*idn?").lower():
                commands = commands_abc + commands_omft
        else:
            print(f"Device Type {deviceType} is unkmown!\nPlease check if the variable 'deviceType' of the Configuration section is configured correctly!")
        # run all commands
        for command in commands:
            if 'sleep' in command:
                try:
                    sleep_time = [int(s) for s in command.split() if s.isdigit()][0]
                except:
                    sleep_time = 5
                print(f"sleep for {sleep_time} seconds")
                sleep(sleep_time)
            else:
                idp_device.sendScpi(command)
    except Exception as e:
        print(f"The following Exception occurred: {e}")
        if "not powered up" in str(e):
            print(f"Please power up the laser cards by pushing the corresponding button or by sending 'pass IDP' (to access user level 1) and afterwards 'powe 1' via SCPI to your device.\n"+
            "You can send SCPI commands using Putty or using the SCPI Control section of the Connection Tab of the device's web gui!")

run_demo()


