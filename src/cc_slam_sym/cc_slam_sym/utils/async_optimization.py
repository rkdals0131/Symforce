#!/usr/bin/env python3
"""
Asynchronous optimization thread for SLAM backend
Prevents blocking the main processing loop
"""

import threading
import queue
import time
from typing import Optional, Tuple
import gtsam
from dataclasses import dataclass

@dataclass
class OptimizationRequest:
    """Request for optimization with current graph state"""
    graph: gtsam.NonlinearFactorGraph
    values: gtsam.Values
    timestamp: float
    request_id: int

@dataclass 
class OptimizationResult:
    """Result from optimization"""
    optimized_values: gtsam.Values
    success: bool
    error: float
    computation_time: float
    request_id: int

class AsyncOptimizer:
    """Runs SLAM optimization in separate thread"""
    
    def __init__(self, max_queue_size: int = 5, logger=None):
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self.logger = logger
        
        self.running = True
        self.thread = threading.Thread(target=self._optimization_loop)
        self.thread.daemon = True
        self.thread.start()
        
        self.request_counter = 0
        self.stats = {
            "total_optimizations": 0,
            "successful": 0,
            "failed": 0,
            "avg_time": 0.0,
            "dropped_requests": 0
        }
        
    def request_optimization(self, graph: gtsam.NonlinearFactorGraph, 
                           values: gtsam.Values) -> bool:
        """Request optimization (non-blocking)"""
        if self.request_queue.full():
            # Drop oldest request if queue full
            try:
                old_req = self.request_queue.get_nowait()
                self.stats["dropped_requests"] += 1
                if self.logger:
                    self.logger.warning(f"[ASYNC_OPT] Dropped request {old_req.request_id}")
            except queue.Empty:
                pass
                
        request = OptimizationRequest(
            graph=graph,
            values=values,
            timestamp=time.time(),
            request_id=self.request_counter
        )
        self.request_counter += 1
        
        try:
            self.request_queue.put_nowait(request)
            return True
        except queue.Full:
            return False
            
    def get_result(self, timeout: float = 0.0) -> Optional[OptimizationResult]:
        """Get optimization result if available"""
        try:
            if timeout > 0:
                return self.result_queue.get(timeout=timeout)
            else:
                return self.result_queue.get_nowait()
        except queue.Empty:
            return None
            
    def _optimization_loop(self):
        """Worker thread for optimization"""
        while self.running:
            try:
                # Wait for request
                request = self.request_queue.get(timeout=0.1)
                
                if self.logger:
                    self.logger.debug(f"[ASYNC_OPT] Processing request {request.request_id}")
                    
                # Run optimization
                start_time = time.time()
                try:
                    optimizer_params = gtsam.LevenbergMarquardtParams()
                    optimizer_params.setVerbosity("SILENT")
                    optimizer_params.setMaxIterations(50)  # Reduced for speed
                    optimizer_params.setRelativeErrorTol(1e-4)
                    optimizer_params.setAbsoluteErrorTol(1e-4)
                    
                    optimizer = gtsam.LevenbergMarquardtOptimizer(
                        request.graph, request.values, optimizer_params
                    )
                    optimized = optimizer.optimize()
                    
                    final_error = request.graph.error(optimized)
                    computation_time = time.time() - start_time
                    
                    result = OptimizationResult(
                        optimized_values=optimized,
                        success=True,
                        error=final_error,
                        computation_time=computation_time,
                        request_id=request.request_id
                    )
                    
                    self.stats["successful"] += 1
                    
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[ASYNC_OPT] Optimization failed: {e}")
                        
                    result = OptimizationResult(
                        optimized_values=request.values,  # Return original
                        success=False,
                        error=float('inf'),
                        computation_time=time.time() - start_time,
                        request_id=request.request_id
                    )
                    
                    self.stats["failed"] += 1
                    
                # Update stats
                self.stats["total_optimizations"] += 1
                self.stats["avg_time"] = (
                    (self.stats["avg_time"] * (self.stats["total_optimizations"] - 1) + 
                     result.computation_time) / self.stats["total_optimizations"]
                )
                
                # Put result
                if self.result_queue.full():
                    # Drop oldest result
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                        
                self.result_queue.put(result)
                
                if self.logger:
                    self.logger.debug(f"[ASYNC_OPT] Completed request {request.request_id} "
                                    f"in {result.computation_time*1000:.1f}ms")
                    
            except queue.Empty:
                # No requests, continue
                pass
            except Exception as e:
                if self.logger:
                    self.logger.error(f"[ASYNC_OPT] Thread error: {e}")
                    
    def shutdown(self):
        """Stop optimization thread"""
        self.running = False
        self.thread.join()