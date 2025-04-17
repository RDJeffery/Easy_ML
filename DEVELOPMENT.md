# Development Notes

**Note:** This is a living document and will be updated as the project evolves.

This document contains notes and guidelines relevant to the development of the Neural Net Playground application.

## Threading with QThread and Workers

When implementing long-running background tasks (like model training) to keep the UI responsive, we use the standard Qt pattern of moving a `QObject` worker to a `QThread`.

**Key Pattern Implementation (`ui_main.py` - `start_training`):**

1.  Create `QThread` instance.
2.  Create `Worker` instance (e.g., `TrainingWorker`) inheriting from `QObject`.
3.  Move the worker to the thread: `worker.moveToThread(thread)`.
4.  **Crucially, connect the necessary signals/slots:**
    *   `thread.started.connect(worker.run)`: Start the worker's task when the thread begins its event loop.
    *   `worker.finished.connect(thread.quit)`: **Essential!** Tells the thread's event loop to exit when the worker's task is done. This is necessary for the thread's `finished` signal to be emitted reliably.
    *   `worker.progress.connect(ui_update_slot)`: Update UI based on worker progress.
    *   `worker.finished.connect(ui_results_slot)`: Handle results from the worker in the UI thread.
    *   `thread.finished.connect(ui_cleanup_slot)`: Perform final UI cleanup/reset *after* the thread has fully stopped (e.g., `_on_thread_actually_finished`).
    *   `thread.finished.connect(worker.deleteLater)`: Schedule the worker object for deletion by Qt's event loop.
    *   `thread.finished.connect(thread.deleteLater)`: Schedule the thread object itself for deletion.
5.  Start the thread: `thread.start()`.

**Why this is important:**

*   Ensures the background task doesn't block the main UI thread.
*   Guarantees proper cleanup of thread and worker objects, preventing memory leaks and crashes (like the "QThread: Destroyed while thread is still running" error).
*   Provides a clear mechanism for communicating progress and results back to the UI.

**Remember to include *all* these connections, especially `worker.finished.connect(thread.quit)`, when adding new background tasks using this pattern.** 