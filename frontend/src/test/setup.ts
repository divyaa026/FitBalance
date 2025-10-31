import '@testing-library/jest-dom';
import { expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import * as matchers from '@testing-library/jest-dom/matchers';

expect.extend(matchers);

afterEach(() => {
<<<<<<< HEAD
  cleanup();
});
=======
    cleanup();
});
>>>>>>> 633c84e602780eab5038f97c9beaa390e270d288
