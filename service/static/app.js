const FEATURES = [
  "AF3",
  "AF4",
  "F3",
  "F4",
  "F7",
  "F8",
  "FC5",
  "FC6",
  "T7",
  "T8",
  "P7",
  "P8",
  "O1",
  "O2",
];

const SAMPLE = {
  AF3: 4329.23,
  AF4: 4393.85,
  F3: 4289.23,
  F4: 4280.51,
  F7: 4009.23,
  F8: 4635.9,
  FC5: 4148.21,
  FC6: 4211.28,
  T7: 4350.26,
  T8: 4238.46,
  P7: 4586.15,
  P8: 4222.05,
  O1: 4096.92,
  O2: 4641.03,
};

const form = document.getElementById("predict-form");
const result = document.getElementById("result");
const fillBtn = document.getElementById("fill-sample");

function createFields() {
  FEATURES.forEach((feature) => {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    label.textContent = feature;
    label.setAttribute("for", feature);

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.id = feature;
    input.name = feature;
    input.placeholder = "e.g. 4300.5";

    wrapper.appendChild(label);
    wrapper.appendChild(input);
    form.appendChild(wrapper);
  });
}

function buildPayload() {
  const payload = {};
  for (const feature of FEATURES) {
    const value = document.getElementById(feature).value;
    payload[feature] = Number.parseFloat(value);
  }
  return payload;
}

function fillSample() {
  for (const feature of FEATURES) {
    document.getElementById(feature).value = SAMPLE[feature];
  }
}

async function submitForm(event) {
  event.preventDefault();
  result.textContent = "Running prediction...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });

    const data = await response.json();
    if (!response.ok) {
      result.textContent = data.error || "Something went wrong.";
      return;
    }

    const prob = (data.eye_closed_probability * 100).toFixed(2);
    const label = data.prediction === 1 ? "Closed" : "Open";
    result.textContent = `Prediction: ${label} (closed prob: ${prob}%)`;
  } catch (err) {
    result.textContent = "Failed to reach the service.";
  }
}

createFields();
fillSample();
form.addEventListener("submit", submitForm);
fillBtn.addEventListener("click", fillSample);
