def calculate_management_index(
    resource_availability,
    shelter_capacity,
    evacuation_rate,
    response_time
):
    response_performance = max(
        0,
        100 - (response_time / 120 * 100)
    )

    management_index = (
        resource_availability * 0.30
        + shelter_capacity * 0.25
        + evacuation_rate * 0.25
        + response_performance * 0.20
    )

    return management_index, response_performance


# Test values
resource_availability = 50
shelter_capacity = 50
evacuation_rate = 50
response_time = 60

management_index, response_performance = calculate_management_index(
    resource_availability,
    shelter_capacity,
    evacuation_rate,
    response_time
)

print("\n================================")
print("   DISASTER MANAGEMENT INDEX")
print("================================")

print(f"Resource Availability: {resource_availability}%")
print(f"Shelter Capacity: {shelter_capacity}%")
print(f"Evacuation Rate: {evacuation_rate}%")
print(f"Response Performance: {response_performance:.2f}%")

print("--------------------------------")
print(f"Management Index: {management_index:.2f}%")
print("================================")