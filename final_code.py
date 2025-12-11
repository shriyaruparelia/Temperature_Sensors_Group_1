import board
import analogio
import time
import busio
import digitalio
import math
from adafruit_onewire.bus import OneWireBus
from adafruit_ds18x20 import DS18X20
import adafruit_max31855
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

#initialising pins for SPI
spi = busio.SPI(clock=board.SCK, MOSI=board.MOSI, MISO=board.MISO)

#initialising chip select pin thermocouple
cs_1 = digitalio.DigitalInOut(board.IO10)
max31855 = adafruit_max31855.MAX31855(spi, cs_1)

#initialising chip select pin for adc
cs_2 = digitalio.DigitalInOut(board.IO8)

#setting pins for analogue sensor
TMP36_PIN = board.IO1
tmp36 = analogio.AnalogIn(TMP36_PIN)

#setting pins for thermistor
NTC_PIN = board.IO0
therm_power = digitalio.DigitalInOut(board.IO9)
therm_power.direction = digitalio.Direction.OUTPUT
reading_ntc = analogio.AnalogIn(NTC_PIN)

#initialising onewire for digital sensor
ow_bus = OneWireBus(board.IO2)
devices = ow_bus.scan() #scan for devices on the OneWire bus
ds18 = DS18X20(ow_bus, ow_bus.scan()[0]) #select first device found

#Function to read temperature of analogue sensor
def tmp36_temperature_C(analogin):
    millivolts = analogin.value * (analogin.reference_voltage * 1000 / 65535)
    return (millivolts - 500) / 10

#Function to read resistance of NTC thermistor
def res_ntc(analogin):
    voltage = analogin.value
    resistance = (10000 *voltage)/(65535  - voltage)
    return resistance

#Function to read temperature of NTC thermistor from resistance
def temp_ntc(resistance):
    temp = (1/(1/298.15 + (1/3950)*math.log(resistance/10000))) - 273
    return temp

print("Time, TMP36, Thermistor, Thermocouple, Digital")

# Main loop
while True:
    #reading time
    t = time.localtime()       # returns a struct_time
    hour = t.tm_hour
    minute = t.tm_min
    second = t.tm_sec
    millisecond = time.monotonic_ns() // 1_000_000 % 1000
    #read temperature from analogue sensor
    therm_power.value = True
    temp_analog = tmp36_temperature_C(tmp36)
    
    #read resistance and temperature of thermistor
    res_NTC = res_ntc(reading_ntc)
    temp_NTC = temp_ntc(res_NTC)
    
    #read temperature of digital sensor
    temp_digital = ds18.temperature
    
    #read temperature of thermocouple
    temp_TC = max31855.temperature
    
    #reading values from adc
#     mcp = MCP.MCP3008(spi, cs_2)
#     channel_0 = AnalogIn(mcp, MCP.P0)
#     channel_4 = AnalogIn(mcp, MCP.P4)
#     adc_analogue = tmp36_temperature_C(channel_4)
#     adc_res_NTC = res_ntc(channel_0)
#     adc_temp_NTC = temp_ntc(adc_res_NTC)
    
    #printing all temperatures
    print(f"{hour:02}:{minute:02}:{second:02}:{millisecond:03d} ,", temp_analog, ",", temp_NTC, ",", temp_TC, ",", temp_digital)
    time.sleep(0.5)
