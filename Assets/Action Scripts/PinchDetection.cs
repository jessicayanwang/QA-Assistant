using UnityEngine;
using UnityEngine.InputSystem.EnhancedTouch;

public class PinchDetection : MonoBehaviour
{
    private bool isUpdating = false;
    public TextUpdater textUpdater;

    private void OnEnable()
    {
        EnhancedTouchSupport.Enable();
    }

    private void OnDisable()
    {
        EnhancedTouchSupport.Disable();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.P))
        {
            // If not already updating text, start looking for updates
            if (!isUpdating)
            {
                isUpdating = true;
                textUpdater.StartUpdatingText();
            }
            // If already looking for text updates, stop looking for updates
            else {
                isUpdating = false;
                textUpdater.StopUpdatingText();
            }
        }
    }
}
