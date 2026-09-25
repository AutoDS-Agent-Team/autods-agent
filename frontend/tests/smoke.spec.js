import { expect, test } from '@playwright/test'

test('authentication shell renders without exposing application internals', async ({ page }) => {
  await page.goto('/')
  await expect(page).toHaveTitle('AutoDS-Agent')
  await expect(page.getByRole('heading', { name: /Enter your data-science workspace/i })).toBeVisible()
  await expect(page.getByPlaceholder('you@example.com')).toBeVisible()
  await expect(page.getByText(/AutoDS-Agent/).first()).toBeVisible()
})
