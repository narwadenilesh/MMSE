"use client";

import * as React from "react";

import { NavMain } from "@/components/nav-main";
import { NavUser } from "@/components/nav-user";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarRail,
} from "@/components/ui/sidebar";
import { Search, Settings2Icon, HistoryIcon } from "lucide-react";

const data = {
  user: {
    name: "Fashion Searcher",
    email: "search@fashion.local",
    avatar: "/avatars/shadcn.jpg",
  },
  navMain: [
    {
      title: "Search",
      url: "#",
      icon: Search,
      isActive: true,
      items: [
        {
          title: "Text Search",
          url: "#text",
        },
        {
          title: "Image Search",
          url: "#image",
        },
        {
          title: "Voice Search",
          url: "#voice",
        },
      ],
    },
    {
      title: "History",
      url: "#",
      icon: HistoryIcon,
      items: [
        {
          title: "Recent Searches",
          url: "#history",
        },
        {
          title: "Saved Items",
          url: "#saved",
        },
      ],
    },
    {
      title: "Settings",
      url: "#",
      icon: Settings2Icon,
      items: [
        {
          title: "Search Settings",
          url: "#settings",
        },
        {
          title: "Preferences",
          url: "#preferences",
        },
      ],
    },
  ],
};

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader>
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <Search className="h-5 w-5 text-primary" />
          </div>
          <div className="flex flex-col gap-0.5 leading-none">
            <span className="font-semibold">Fashion Search</span>
            <span className="text-xs text-muted-foreground">
              Vector-Powered
            </span>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
