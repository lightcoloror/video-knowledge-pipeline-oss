// DOM shims for the same jsdom-based editor test approach used upstream.
if (!Range.prototype.getClientRects) Range.prototype.getClientRects = () => [] as any;
if (!Range.prototype.getBoundingClientRect) Range.prototype.getBoundingClientRect = () => ({ x: 0, y: 0, width: 0, height: 0, top: 0, left: 0, right: 0, bottom: 0, toJSON() {} }) as any;
