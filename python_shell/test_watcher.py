#!/usr/bin/env python3
"""
Test script for FileWatcher functionality.
Run with: python python_shell/test_watcher.py
"""

import time
import os
from pathlib import Path
import sys
import shutil

# Add the project root to the python path so it can find the semantic_engine
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

try:
    from atlas.semantic_engine import FileWatcher
except ImportError as e:
    print("="*80)
    print("ERROR: Could not import the `semantic_engine` module.")
    print("Please make sure you have built the Rust core.")
    print("From the `rust_core` directory, run: `maturin develop`")
    print(f"Original error: {e}")
    print("="*80)
    sys.exit(1)


def test_basic_watching():
    """Test basic file watching"""
    # Create temp directory
    watch_dir = Path("./temp_watch_test")
    watch_dir.mkdir(exist_ok=True)
    test_file = None

    print(f"Watching: {watch_dir.resolve()}\n")
    
    # Create watcher
    watcher = FileWatcher(str(watch_dir.resolve()))
    
    try:
        # Test 1: Create file
        print("Test 1: Creating file...")
        test_file = watch_dir / "test.py"
        test_file.write_text("# Initial content")
        time.sleep(0.3)
        
        events = watcher.poll_events()
        print(f"  Events: {[repr(e) for e in events]}")
        assert any(e.event_type == "created" for e in events), "Should detect creation"
        
        # Clear events before next test
        watcher.poll_events()

        # Test 2: Modify file
        print("\nTest 2: Modifying file...")
        test_file.write_text("# Modified content")
        time.sleep(0.3)
        
        events = watcher.poll_events()
        print(f"  Events: {[repr(e) for e in events]}")
        assert any(e.event_type == "modified" for e in events), "Should detect modification"
        
        # Test 3: Delete file
        print("\nTest 3: Deleting file...")
        test_file.unlink()
        time.sleep(0.3)
        
        events = watcher.poll_events()
        print(f"  Events: {[repr(e) for e in events]}")
        assert any(e.event_type == "deleted" for e in events), "Should detect deletion"
        
        # Test 4: Statistics
        print("\nTest 4: Getting statistics...")
        stats = watcher.get_stats()
        print(f"  Stats: {stats}")
        assert stats.events_received > 0, "Should have received events"
        
        print("\n✅ All tests passed!")
        
    finally:
        print("\nStopping watcher...")
        watcher.stop()
        
        # Cleanup
        if test_file and test_file.exists():
            test_file.unlink()
        if watch_dir.exists():
            shutil.rmtree(watch_dir)

def test_context_manager():
    """Tests that the watcher can be created and stopped."""
    watch_dir = Path("./temp_watch_test2")
    watch_dir.mkdir(exist_ok=True)
    watcher = None
    
    test_file = watch_dir / "test.py"
    try:
        print("\nTesting watcher creation and teardown...")
        
        watcher = FileWatcher(str(watch_dir))
        
        # Create file
        test_file.write_text("content")
        time.sleep(0.3)
        
        events = watcher.poll_events()
        print(f"  Detected {len(events)} events")
        assert any('test.py' in e.path for e in events)
            
        print("✅ Watcher created and polled successfully!")
        
    finally:
        # Cleanup
        if watcher:
            watcher.stop()
        if test_file.exists():
            test_file.unlink()
        if watch_dir.exists():
            shutil.rmtree(watch_dir)

def test_filtering():
    """Test file filtering"""
    watch_dir = Path("./temp_watch_test3")
    watch_dir.mkdir(exist_ok=True)
    
    py_file = watch_dir / "test.py"
    txt_file = watch_dir / "test.txt"
    try:
        print("\nTesting file filtering...")
        
        watcher = FileWatcher(
            str(watch_dir),
            extensions=["py"],  # Only watch Python files
        )
        
        # Create Python file (should be detected)
        py_file.write_text("# Python")
        
        # Create text file (should be filtered)
        txt_file.write_text("text")
        
        time.sleep(0.3)
        
        events = watcher.poll_events()
        print(f"  Events: {[e.path for e in events]}")
        
        # Should only see Python file
        py_events = [e for e in events if "test.py" in e.path]
        txt_events = [e for e in events if "test.txt" in e.path]
        
        assert len(py_events) > 0, "Should detect Python file"
        assert len(txt_events) == 0, "Should not detect txt file"
        
        watcher.stop()
        
        print("✅ Filtering test passed!")
        
    finally:
        # Cleanup
        if py_file.exists():
            py_file.unlink()
        if txt_file.exists():
            txt_file.unlink()
        watch_dir.rmdir()

if __name__ == "__main__":
    print("="* 60)
    print("FileWatcher Integration Tests")
    print("="* 60)
    
    test_basic_watching()
    test_context_manager()
    test_filtering()
    
    print("\n" + "="* 60)
    print("All tests completed successfully! ✅")
    print("="* 60)
