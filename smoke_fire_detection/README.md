# Smoke & Fire KPI

`Salahyolo26.pt` is loaded once and evaluated through the application's shared GPU lock. The package filters detections with the existing camera polygon ROI, associates nearby smoke/fire detections into bounded camera-local incidents, confirms them over time, and returns only newly confirmed incidents to the existing event infrastructure.

Set `SMOKE_FIRE_TEST_MODE=true` to retain annotation and state-transition logs without publishing alerts.
