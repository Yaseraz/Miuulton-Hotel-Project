function calculateNumberOfNights() {
    var reservationDate = new Date(document.getElementById("reservation_date").value);
    var checkoutDate = new Date(document.getElementById("checkout_date").value);

    // Check if the Check-Out date is before the Check-In date
    if (checkoutDate.getTime() < reservationDate.getTime()) {
        alert("Check-Out date cannot be earlier than Check-In date.");
        document.getElementById("checkout_date").value = "";
        return;
    }

    // Calculate the number of nights between the two dates
    var timeDiff = checkoutDate.getTime() - reservationDate.getTime();
    var diffDays = Math.ceil(timeDiff / (1000 * 3600 * 24));

    // Calculate the number of weekdays and weekends in the interval
    var numWeekdays = 0;
    var numWeekends = 0;
    var currentDate = new Date(reservationDate.getTime());
    for (var i = 0; i < diffDays; i++) {
        if (currentDate.getDay() == 0 || currentDate.getDay() == 6) {
            numWeekends++;
        } else {
            numWeekdays++;
        }
        currentDate.setDate(currentDate.getDate() + 1);
    }

    // Set the number of nights and weekends on the form
    document.getElementById("no_of_weekend_nights").value = numWeekends;
    document.getElementById("no_of_week_nights").value = numWeekdays;
}


function calculateLeadTime() {
  const reservationDateInput = document.getElementById("reservation_date");
  const selectedDate = new Date(reservationDateInput.value);
  const today = new Date();

  const leadTime = Math.ceil((selectedDate.getTime() - today.getTime()) / (1000 * 3600 * 24));
  document.getElementById("lead_time").value = leadTime;

  const dayInput = document.querySelector("#arrival_date");
  const monthInput = document.querySelector("#arrival_month");
  const yearInput = document.querySelector("#arrival_year");

  dayInput.value = selectedDate.getDate();
  monthInput.value = selectedDate.getMonth() + 1;
  yearInput.value = selectedDate.getFullYear();
}


function showLogin() {
    var loginDiv = document.getElementById("login");
    var guestSelect = document.getElementById("repeated_guest");
    if (guestSelect.value == "1") {
        loginDiv.style.display = "block";
    } else {
        loginDiv.style.display = "none";
        guestSelect.value = "0";
    }
}

function checkCredentials() {
    var username = document.getElementById("username").value;
    var password = document.getElementById("password").value;
    if (username == "Vahit" && password == "Upsala46") {
        alert("Login successful!");
        document.getElementById("no_of_previous_cancellations").value = 1;
        document.getElementById("no_of_previous_bookings_not_canceled").value = 5;
    } else {
        alert("Invalid username or password!");
        document.getElementById("username").value = "";
        document.getElementById("password").value = "";
        document.getElementById("repeated_guest").value = "0";
        showLogin();
    }
}

function updateRoomPrice() {
    const roomType = document.getElementById("room_type_reserved");
    // Average price input
    const avgPrice = document.getElementById("avg_price_per_room");

    // Define room prices
    const roomPrices = {
      "Room_Type 1": 95,
      "Room_Type 2": 87,
      "Room_Type 3": 73,
      "Room_Type 4": 125,
      "Room_Type 5": 123,
      "Room_Type 6": 182,
      "Room_Type 7": 155,
    };

    // Event listener for room type selection
    roomType.addEventListener("change", () => {
      const selectedRoomType = roomType.value;
      // Update average price input with the corresponding price
      avgPrice.value = roomPrices[selectedRoomType];
    });
    }



function updateSpecialRequestCount() {
  // Get all the checkboxes
  var checkboxes = document.querySelectorAll('input[type="checkbox"]');
  // Initialize the count
  var count = 0;

  // Loop through the checkboxes and add event listeners
  checkboxes.forEach(function(checkbox) {
      checkbox.addEventListener('change', function() {
          count = 0;
          // Loop through all the checkboxes again to count the checked ones
          checkboxes.forEach(function(checkbox) {
              if (checkbox.checked) {
                  count++;
              }
          });

          // Update the value of the count input
          document.getElementById('no_of_special_requests').value = count;
      });
  });
}



