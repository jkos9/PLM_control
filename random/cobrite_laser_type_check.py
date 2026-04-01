import serial


USB_COM_PORT = "COM10"
BAUDRATE = 115200
TIMEOUT_SECONDS = 2


def infer_device_type(idn_reply: str) -> str:
	idn = idn_reply.lower()
	if "dx1" in idn:
		return "DX1"
	if "abc" in idn or "omft" in idn:
		return "ABC/OMFT"
	return "Cobrite without DX1"


def query_idn(com_port: str) -> str:
	with serial.Serial(com_port, BAUDRATE, timeout=TIMEOUT_SECONDS) as ser:
		ser.reset_input_buffer()
		ser.reset_output_buffer()
		ser.write(b"*idn?;")
		reply = ser.read_until(b";").decode("utf-8", errors="replace").strip()
		return reply.rstrip(";")


def main() -> None:
	idn_reply = query_idn(USB_COM_PORT)
	device_type = infer_device_type(idn_reply)
	print(f"IDN reply: {idn_reply}")
	print(f"Use deviceType = \"{device_type}\"")


if __name__ == "__main__":
	main()
